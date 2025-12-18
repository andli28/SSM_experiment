# long-context-ssm/data/longbench_v2_utils.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import json
from datasets import load_dataset
from transformers import AutoTokenizer


# ---------- paths & IO ----------

def get_lbv2_root(prompt_root: Path) -> Path:
    """Subdirectory under PROMPT_ROOT where all LongBench-v2 prompt sets live."""
    root = Path(prompt_root) / "longbench_v2"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_prompt_records(records: List[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def load_prompt_records(path: Path) -> List[Dict]:
    path = Path(path)
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    print(f"Loaded {len(out)} records from {path}")
    return out


# ---------- dataset + tokenizer ----------

def load_longbench_v2(split: str = "train"):
    """Load LongBench-v2 from HuggingFace."""
    ds = load_dataset("THUDM/LongBench-v2", split=split)
    print(ds)
    return ds


def get_lb2_tokenizer(model_id: str):
    """Tokenizer used to measure/construct prompts."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok


# ---------- prompt construction with truncation ----------

def build_truncated_prompt_for_ctx(
    sample: Dict,
    tokenizer,
    target_ctx_len: int,
    max_new_tokens: int,
    safety_margin: int = 32,
) -> Tuple[str, int] | None:
    """
    Build a prompt for a given LongBench-v2 sample that *fits* into
    target_ctx_len (input side), by truncating the context if necessary.

    Returns (prompt_str, n_tokens) or None if we truly cannot fit anything.
    """
    # Pieces of the prompt, separated so we can control context length
    header = (
        "You are a helpful assistant.\n\n"
        "Read the following context and answer the multiple-choice question.\n\n"
        "Context:\n"
    )

    ctx = sample["context"]
    q   = sample["question"]
    A   = sample["choice_A"]
    B   = sample["choice_B"]
    C   = sample["choice_C"]
    D   = sample["choice_D"]

    tail = (
        f"\n\nQuestion:\n{q}\n\n"
        "Choices:\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n\n"
        "Answer with a single letter: A, B, C, or D."
    )

    # Tokenize header, context, and tail separately
    header_ids = tokenizer.encode(header, add_special_tokens=False)
    tail_ids   = tokenizer.encode(tail, add_special_tokens=False)
    ctx_ids    = tokenizer.encode(ctx, add_special_tokens=False)

    hard_max_input = target_ctx_len - max_new_tokens - safety_margin
    usable_for_ctx = hard_max_input - len(header_ids) - len(tail_ids)

    if usable_for_ctx <= 0:
        # the non-context parts already blow the budget – give up on this sample
        return None

    if len(ctx_ids) > usable_for_ctx:
        # truncate context tokens to fit
        ctx_ids = ctx_ids[:usable_for_ctx]

    truncated_ctx = tokenizer.decode(ctx_ids)

    prompt = header + truncated_ctx + tail

    # Optional sanity check
    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n_tokens > hard_max_input:
        # Very defensive: if something went wrong, skip
        print(
            f"[WARN] Prompt still too long after truncation "
            f"(n_tokens={n_tokens}, limit={hard_max_input}). Skipping sample."
        )
        return None

    return prompt, n_tokens


# ---------- main builders / loaders ----------

def build_lb2_prompt_sets(
    prompt_root: Path,
    tokenizer_model_id: str,
    pure_ssm_contexts: List[int],
    max_new_tokens: int,
    split: str = "train",
    max_examples_per_ctx: int = 200,
    tol: int = 512,   # kept for backwards compatibility, unused
) -> None:
    """
    For each context length in pure_ssm_contexts, build a JSONL file
    (<prompt_root>/longbench_v2/lbv2_8k.jsonl etc) where each record is:

        {
          "prompt": <string>,
          "target": "A"/"B"/"C"/"D",
          "dataset": "longbench_v2",
          "split": <split>,
          "example_id": <_id or index>,
          "meta": {...}
        }

    Contexts are TRUNCATED as needed so that the total input tokens fit
    inside target_ctx_len - max_new_tokens - safety_margin.
    """
    prompt_root = Path(prompt_root)
    lbv2_root   = get_lbv2_root(prompt_root)

    ds  = load_longbench_v2(split=split)
    tok = get_lb2_tokenizer(tokenizer_model_id)

    print("Building LB-v2 prompt sets with tokenizer:", tokenizer_model_id)
    print("Contexts:", pure_ssm_contexts, "max_new_tokens:", max_new_tokens)

    for ctx_len in pure_ssm_contexts:
        tag = f"{ctx_len // 1024}k"
        records: List[Dict] = []

        for i, sample in enumerate(ds):
            out = build_truncated_prompt_for_ctx(
                sample=sample,
                tokenizer=tok,
                target_ctx_len=ctx_len,
                max_new_tokens=max_new_tokens,
            )
            if out is None:
                continue

            prompt, n_tokens = out
            rec = {
                "prompt": prompt,
                "target": sample["answer"],          # 'A' / 'B' / 'C' / 'D'
                "dataset": "longbench_v2",
                "split": split,
                "example_id": sample.get("_id", i),
                "meta": {
                    "domain":       sample.get("domain"),
                    "sub_domain":   sample.get("sub_domain"),
                    "difficulty":   sample.get("difficulty"),
                    "length_bin":   sample.get("length"),
                    "prompt_tokens": n_tokens,
                    "target_ctx_len": ctx_len,
                    "max_new_tokens": max_new_tokens,
                },
            }
            records.append(rec)

            if max_examples_per_ctx is not None and len(records) >= max_examples_per_ctx:
                break

        if records:
            lens = [r["meta"]["prompt_tokens"] for r in records]
            print(
                f"[LBv2 ctx={ctx_len}] collected={len(records)}, "
                f"min_len={min(lens)}, max_len={max(lens)}, "
                f"mean_len={sum(lens)/len(lens):.1f}"
            )
        else:
            print(f"[LBv2 ctx={ctx_len}] WARNING: collected 0 records.")

        out_path = lbv2_root / f"lbv2_{tag}.jsonl"
        save_prompt_records(records, out_path)


def load_lb2_prompts_for_tag(
    prompt_root: Path,
    tag: str,
) -> Tuple[List[Dict], List[str]]:
    """
    tag ∈ {"8k", "16k", "32k"}.
    Returns (records, prompts).
    """
    prompt_root = Path(prompt_root)
    lbv2_root   = get_lbv2_root(prompt_root)
    path        = lbv2_root / f"lbv2_{tag}.jsonl"

    records = load_prompt_records(path)
    prompts = [r["prompt"] for r in records]
    return records, prompts
