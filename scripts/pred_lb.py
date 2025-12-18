#!/usr/bin/env python3
"""
LongBench-v2 evaluator using in-process vLLM (LLM.generate) with batching.

- No RAG / CoT modes (intentionally)
- Optional W&B logging (pass --wandb)

"""

import argparse
import json
import os
import re
import statistics
import textwrap
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "scripts" / "prompts"

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

try:
    import pynvml  # type: ignore

    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


DEFAULT_0SHOT = """$DOC$

Question: $Q$

A. $C_A$
B. $C_B$
C. $C_C$
D. $C_D$

Answer:"""

try:
    TEMPLATE_0SHOT = _read(PROMPTS_DIR / "0shot.txt")
except Exception:
    TEMPLATE_0SHOT = DEFAULT_0SHOT


def build_prompt(item: Dict[str, Any]) -> Tuple[str, str]:
    context = item["context"]
    prompt = (
        TEMPLATE_0SHOT.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )
    return prompt, context


ANSWER_PATTERNS = [
    re.compile(r"(?:final\s*answer|answer)\s*[:\-]\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s*\(?([A-D])\)?", re.IGNORECASE),
]


def extract_answer(response: str) -> Optional[str]:
    if not response:
        return None
    text = response.replace("*", "").strip()
    tail = text[-1000:]
    for pat in ANSWER_PATTERNS:
        m = pat.search(tail)
        if m:
            return m.group(1).upper()
    m = re.search(r"(?:^|\n)\s*([A-D])\s*$", text[-10:])
    return m.group(1).upper() if m else None


def _encode(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


def _decode(tok, ids: List[int]) -> str:
    return tok.decode(ids, skip_special_tokens=True)


def truncate_prompt(prompt: str, tok, max_len: int) -> str:
    if max_len <= 0:
        return prompt
    ids = _encode(tok, prompt)
    if len(ids) <= max_len:
        return prompt
    half = max_len // 2
    ids = ids[:half] + ids[-half:]
    return _decode(tok, ids)


def should_use_chat(model_name: str) -> bool:
    s = (model_name or "").lower()
    return any(k in s for k in ["instruct", "chat", "assistant"])


def maybe_apply_chat_template(tok, prompt: str, use_chat: bool) -> str:
    if not use_chat:
        return prompt
    try:
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(  # type: ignore[attr-defined]
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass
    return prompt


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


class VramSampler:
    """Peak VRAM (MiB) across all GPUs via pynvml (optional dependency)."""

    def __init__(self, poll_s: float = 0.1):
        self.poll_s = poll_s
        self.peak_mib_per_gpu_max = 0.0
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._handles: List[Any] = []

    def __enter__(self):
        if not _NVML_OK:
            return self
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not _NVML_OK:
            return
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            mx = 0.0
            for h in self._handles:
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                mx = max(mx, info.used / (1024**2))
            self.peak_mib_per_gpu_max = max(self.peak_mib_per_gpu_max, mx)
            time.sleep(self.poll_s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", "-s", type=str, default=str(ROOT / "results"))
    p.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("HF_DATASETS_CACHE", str(ROOT / "hf_datasets")),
    )
    p.add_argument(
        "--model", "-m", type=str, required=True
    )  # HF id/path for in-process vLLM
    p.add_argument(
        "--served_name", type=str, default=""
    )  # for filenames/W&B (optional)
    p.add_argument(
        "--tokenizer_id", type=str, default=os.environ.get("TOKENIZER_ID", "")
    )
    p.add_argument(
        "--max_len", type=int, default=int(os.environ.get("CTX_LEN", "8192"))
    )
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--rep", type=int, default=int(os.environ.get("RUN_REP", "1")))

    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument(
        "--dtype", type=str, default=os.environ.get("VLLM_DTYPE", "bfloat16")
    )
    p.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "1")))
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")),
    )

    p.add_argument("--force_chat", action="store_true")
    p.add_argument("--force_completion", action="store_true")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def build_llm(args) -> LLM:
    return LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def _chunked(xs: List[Any], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def run_eval(args, llm: Optional[LLM] = None) -> Dict[str, Any]:
    os.makedirs(args.save_dir, exist_ok=True)

    served_name = args.served_name or args.model
    tokenizer_id = args.tokenizer_id or served_name
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    use_chat = bool(args.force_chat) or (
        not args.force_completion and should_use_chat(served_name)
    )

    owned_llm = False
    if llm is None:
        llm = build_llm(args)
        owned_llm = True

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", served_name)
    out_file = os.path.join(
        args.save_dir, f"{safe}__ctx{args.max_len}__rep{args.rep}.jsonl"
    )

    dataset = load_dataset(
        "THUDM/LongBench-v2",
        split="train",
        cache_dir=args.cache_dir,
        download_mode="reuse_dataset_if_exists",
    )

    data = [
        {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        for item in dataset
    ]

    run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed but --wandb was set.")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "longbench-vllm"),
            entity=os.environ.get("WANDB_ENTITY"),
            group=os.environ.get("WANDB_GROUP"),
            name=os.environ.get("WANDB_NAME")
            or f"{safe}__longbench__ctx{args.max_len}__rep{args.rep}",
            config={
                "model/served_name": served_name,
                "model/hf_id": args.model,
                "tokenizer_id": tokenizer_id,
                "protocol/seed": args.seed,
                "protocol/temperature": args.temperature,
                "protocol/top_p": args.top_p,
                "protocol/max_new_tokens": args.max_new_tokens,
                "protocol/batch_size": args.batch_size,
                "ctx_len": args.max_len,
                "rep": args.rep,
                "use_chat_template": use_chat,
                "engine/dtype": args.dtype,
                "engine/tp": args.tp,
                "engine/gpu_memory_utilization": args.gpu_memory_utilization,
            },
        )

    lat_e2e: List[float] = []
    lat_ttft: List[float] = []  # not available in llm.generate
    out_tokens: List[int] = []
    reprompt_used = 0

    counts = {"easy": 0, "hard": 0, "short": 0, "medium": 0, "long": 0, "total": 0}
    correct = {"easy": 0, "hard": 0, "short": 0, "medium": 0, "long": 0, "total": 0}

    # Throughput must use true wall-clock time across batches
    wall_time_total = 0.0

    sp_main = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
    )
    sp_rep = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=2,
        seed=int(args.seed),
    )

    prompt_budget = max(256, args.max_len - args.max_new_tokens - 64)

    with VramSampler(poll_s=0.1) as vs:
        with open(out_file, "a", encoding="utf-8") as fout:
            for batch_items in list(_chunked(data, args.batch_size)):
                prompts: List[str] = []
                contexts: List[str] = []
                for item in batch_items:
                    p_raw, ctx = build_prompt(item)
                    p_raw = truncate_prompt(p_raw, tok, prompt_budget)
                    p_raw += (
                        "\n\nYou MUST end with: Final answer: X (X is A, B, C, or D)."
                    )
                    prompts.append(maybe_apply_chat_template(tok, p_raw, use_chat))
                    contexts.append(ctx)

                t0 = time.perf_counter()
                outs = llm.generate(prompts, sp_main)  # type: ignore[arg-type]
                t1 = time.perf_counter()
                batch_time = t1 - t0
                wall_time_total += batch_time

                texts: List[str] = []
                for o in outs:
                    txt = ""
                    if o.outputs:
                        txt = o.outputs[0].text or ""
                    texts.append(txt)

                preds: List[Optional[str]] = [
                    extract_answer((t or "").strip()) for t in texts
                ]

                need_rep = [i for i, p in enumerate(preds) if p is None]
                rep_texts: Dict[int, str] = {}
                rep_batch_time = 0.0

                if need_rep:
                    rep_prompts: List[str] = []
                    for i in need_rep:
                        item = batch_items[i]
                        prev = (texts[i] or "").strip()
                        reprompt = textwrap.dedent(f"""\
                            Return ONE letter on a single line: A, B, C, or D.

                            Q: {item["question"].strip()}
                            Prev: {prev}
                            A: {item["choice_A"].strip()}
                            B: {item["choice_B"].strip()}
                            C: {item["choice_C"].strip()}
                            D: {item["choice_D"].strip()}

                            Final answer:""")
                        rep_prompts.append(
                            maybe_apply_chat_template(tok, reprompt, use_chat)
                        )

                    t2 = time.perf_counter()
                    rep_outs = llm.generate(rep_prompts, sp_rep)  # type: ignore[arg-type]
                    t3 = time.perf_counter()
                    rep_batch_time = t3 - t2
                    wall_time_total += rep_batch_time
                    reprompt_used += len(need_rep)

                    for j, o in enumerate(rep_outs):
                        txt = ""
                        if o.outputs:
                            txt = o.outputs[0].text or ""
                        rep_texts[need_rep[j]] = txt

                    for i in need_rep:
                        pred2 = extract_answer((rep_texts.get(i, "") or "").strip())
                        if pred2 is not None:
                            preds[i] = pred2

                for i, item in enumerate(batch_items):
                    response = (texts[i] or "").strip()
                    if i in rep_texts:
                        response = response + "\n\n[REPROMPT]\n" + (rep_texts[i] or "")

                    pred = preds[i]
                    judge = pred == item["answer"]

                    try:
                        n_out = len(_encode(tok, response))
                    except Exception:
                        n_out = 0

                    e2e_s = batch_time + (rep_batch_time if i in rep_texts else 0.0)
                    lat_e2e.append(float(e2e_s))
                    lat_ttft.append(float("nan"))
                    out_tokens.append(int(n_out))

                    counts["total"] += 1
                    correct["total"] += int(judge)

                    d = (
                        item["difficulty"]
                        if item["difficulty"] in ("easy", "hard")
                        else "hard"
                    )
                    counts[d] += 1
                    correct[d] += int(judge)

                    L = (
                        item["length"]
                        if item["length"] in ("short", "medium", "long")
                        else "long"
                    )
                    counts[L] += 1
                    correct[L] += int(judge)

                    item_out = dict(item)
                    item_out["response"] = response
                    item_out["pred"] = pred
                    item_out["judge"] = judge
                    item_out["context"] = contexts[i][:1000]
                    fout.write(json.dumps(item_out, ensure_ascii=False) + "\n")

                fout.flush()

                if run is not None and (counts["total"] % args.log_every == 0):
                    wandb.log(
                        {
                            "progress/seen": counts["total"],
                            "eff/latency_e2e_s_mean_sofar": float(
                                statistics.mean(lat_e2e)
                            )
                            if lat_e2e
                            else float("nan"),
                            "eff/ttft_s_mean_sofar": float("nan"),
                            "debug/reprompt_used_sofar": reprompt_used,
                        }
                    )

    def acc(k: str) -> float:
        return (correct[k] / counts[k]) if counts[k] > 0 else float("nan")

    e2e_mean = float(statistics.mean(lat_e2e)) if lat_e2e else float("nan")
    e2e_p50 = float(percentile(lat_e2e, 0.50))
    e2e_p95 = float(percentile(lat_e2e, 0.95))

    total_out = float(sum(out_tokens))
    toks_per_s = (total_out / wall_time_total) if wall_time_total > 0 else float("nan")
    peak_vram = float(getattr(vs, "peak_mib_per_gpu_max", float("nan")))

    summary = {
        "acc_overall": acc("total"),
        "acc_easy": acc("easy"),
        "acc_hard": acc("hard"),
        "acc_short": acc("short"),
        "acc_medium": acc("medium"),
        "acc_long": acc("long"),
        "e2e_mean_s": e2e_mean,
        "e2e_p50_s": e2e_p50,
        "e2e_p95_s": e2e_p95,
        "ttft_mean_s": float("nan"),
        "ttft_p50_s": float("nan"),
        "ttft_p95_s": float("nan"),
        "tokens_per_s": toks_per_s,
        "peak_vram_mib": peak_vram,
        "reprompt_used": reprompt_used,
        "out_file": out_file,
        "wall_time_s": float(wall_time_total),
    }

    print(json.dumps(summary, indent=2))

    if run is not None:
        wandb.log(
            {
                "acc/overall": summary["acc_overall"],
                "acc/easy": summary["acc_easy"],
                "acc/hard": summary["acc_hard"],
                "acc/short": summary["acc_short"],
                "acc/medium": summary["acc_medium"],
                "acc/long": summary["acc_long"],
                "eff/latency_e2e_s_mean": e2e_mean,
                "eff/latency_e2e_s_p50": e2e_p50,
                "eff/latency_e2e_s_p95": e2e_p95,
                "eff/output_tokens_total": total_out,
                "eff/tokens_per_s": toks_per_s,
                "eff/peak_vram_mib_per_gpu_max": peak_vram,
                "debug/reprompt_used_total": reprompt_used,
            }
        )
        run.finish()

    if owned_llm:
        try:
            del llm
        except Exception:
            pass

    return summary


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
