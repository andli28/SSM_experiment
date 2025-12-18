# eval_longbench_mc.py
import os
import argparse
import gc
from typing import List, Tuple

from pathlib import Path

import torch

# Let vLLM / HF download models if needed (not forced offline)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

REPO_ROOT = Path(__file__).resolve().parent

from runner_ssm import (
    load_llm,
    choose_mc_option,
    make_run_configs_for_pure_ssm,
)
from data_loader import (
    LB_V2_PROMPTS_BY_CTX,
    LB_V2_LABELS_BY_CTX,
)


def filter_lb_prompts_for_model(
    llm,
    prompts: List[str],
    labels: List[str],
    ctx_len: int,
    *,
    answer_prefix: str = " ",
    options=("A", "B", "C", "D"),
    safety_margin_tokens: int = 16,
) -> Tuple[List[str], List[str]]:
    """
    Re-filter LongBench prompts so that, under THIS model's tokenizer,
    [prompt + answer_prefix + option] always fits inside ctx_len tokens.

    Returns (filtered_prompts, filtered_labels).
    """
    tok = llm.get_tokenizer()

    # Max tokens taken by " answer" where answer is one of A/B/C/D
    max_opt_tokens = 0
    for opt in options:
        opt_ids = tok.encode(answer_prefix + opt, add_special_tokens=False)
        max_opt_tokens = max(max_opt_tokens, len(opt_ids))

    hard_max_prompt_tokens = ctx_len - max_opt_tokens - safety_margin_tokens

    keep_prompts: List[str] = []
    keep_labels: List[str] = []
    dropped = 0

    for p, y in zip(prompts, labels):
        n = len(tok.encode(p, add_special_tokens=False))
        if n <= hard_max_prompt_tokens:
            keep_prompts.append(p)
            keep_labels.append(y)
        else:
            dropped += 1

    print(
        f"[filter_lb_prompts_for_model] ctx_len={ctx_len}, "
        f"hard_max_prompt_tokens={hard_max_prompt_tokens} -> "
        f"kept {len(keep_prompts)} / {len(prompts)} prompts, "
        f"dropped {dropped} too-long prompts."
    )
    return keep_prompts, keep_labels


def find_run_config(model_key: str, ctx_len: int):
    """
    Grab the matching RunConfig from make_run_configs_for_pure_ssm().
    """
    cfgs = make_run_configs_for_pure_ssm()
    matches = [
        c for c in cfgs
        if c.model_key == model_key and c.context_len == ctx_len
    ]
    if not matches:
        raise ValueError(
            f"No RunConfig found for model_key={model_key}, ctx_len={ctx_len}."
        )
    if len(matches) > 1:
        print(
            f"[WARN] Multiple RunConfigs for model_key={model_key}, "
            f"ctx_len={ctx_len}. Using the first one."
        )
    return matches[0]


def run_eval_for_ctx(model_key: str, ctx_len: int):
    """
    Run LongBench v2 MC evaluation for a single (model_key, ctx_len) pair.
    Writes predictions to longbench_mc_preds/lbv2_mc_{model_key}_{ctx_len}.jsonl
    and prints accuracy.
    """
    if ctx_len not in LB_V2_PROMPTS_BY_CTX:
        print(
            f"[SKIP] No LongBench v2 prompts loaded for ctx_len={ctx_len}. "
            "Make sure lbv2_{8k,16k,32k}.jsonl exist in data/prompt_sets/longbench_v2/."
        )
        return

    prompts_all = LB_V2_PROMPTS_BY_CTX[ctx_len]
    labels_all = LB_V2_LABELS_BY_CTX[ctx_len]
    assert len(prompts_all) == len(labels_all)

    print(
        f"\n=== LongBench MC eval | model_key={model_key} @ ctx_len={ctx_len} "
        f"(num_raw_prompts={len(prompts_all)}) ==="
    )

    # 1) Get the proper RunConfig and load the vLLM model
    cfg = find_run_config(model_key, ctx_len)
    llm = load_llm(cfg)

    # 2) Filter prompts for this model + context length
    prompts, gold_labels = filter_lb_prompts_for_model(
        llm,
        prompts_all,
        labels_all,
        ctx_len=ctx_len,
        answer_prefix=" ",
        options=("A", "B", "C", "D"),
        safety_margin_tokens=16,
    )

    # 3) Score each prompt with choose_mc_option
    n_correct = 0
    preds: List[str] = []

    for i, (prompt, gold) in enumerate(zip(prompts, gold_labels)):
        pred = choose_mc_option(
            llm,
            prompt,
            options=["A", "B", "C", "D"],
        )
        preds.append(pred)
        if pred == gold:
            n_correct += 1

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(prompts)} prompts...")

    acc = n_correct / len(prompts) if prompts else 0.0
    print(
        f"Accuracy for {model_key} @ {ctx_len} tokens "
        f"({len(prompts)} prompts): {acc:.3%}"
    )

    # 4) Save predictions
    lb_mc_root = REPO_ROOT / "longbench_mc_preds"
    lb_mc_root.mkdir(parents=True, exist_ok=True)
    out_path = lb_mc_root / f"lbv2_mc_{model_key}_{ctx_len}.jsonl"

    import json
    with out_path.open("w", encoding="utf-8") as f:
        for i, (prompt, gold, pred) in enumerate(zip(prompts, gold_labels, preds)):
            f.write(json.dumps({
                "model_key": model_key,
                "context_len": ctx_len,
                "prompt_idx": i,
                "gold": gold,
                "pred": pred,
            }) + "\n")

    print("Saved predictions to", out_path)

    # 5) Clean up CUDA memory
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        choices=["mamba-2.8b", "mamba-codestral-7b"],
        required=True,
        help="Which pure SSM model to evaluate.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        nargs="+",
        default=[8192],
        help="One or more context lengths to evaluate, "
             "e.g. --ctx 8192 16384 32768",
    )
    args = parser.parse_args()

    target_ctxs = sorted(set(args.ctx))
    for ctx_len in target_ctxs:
        run_eval_for_ctx(args.model_key, ctx_len)


if __name__ == "__main__":
    main()
