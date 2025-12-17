#!/usr/bin/env python3
"""
Ada-LEval runner using in-process vLLM (LLM.generate) with batching.

Assumes:
- data in:  <repo_root>/data
- code in:  <repo_root>/ada_leval
- this file: <repo_root>/scripts/pred_leval.py
"""

import argparse
import json
import os
import re
import statistics
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams  # type: ignore

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from ada_leval.dataset import StackSelect, TextSort  # noqa: E402
from ada_leval.util import dump, load  # noqa: E402

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

try:
    import pynvml  # type: ignore

    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _parse_k(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(s)


TEXTSORT_SETTINGS = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
STACKSELECT_SETTINGS = [
    "1k",
    "2k",
    "4k",
    "6k",
    "8k",
    "12k",
    "16k",
    "32k",
    "64k",
    "128k",
]


def pick_setting_for_ctx(ctx_tokens: int, settings: List[str]) -> str:
    vals = sorted((_parse_k(x), x) for x in settings)
    best = vals[0][1]
    for v, name in vals:
        if v <= ctx_tokens:
            best = name
        else:
            break
    return best


def auto_datasets_for_ctx(ctx_tokens: int) -> List[str]:
    ts = pick_setting_for_ctx(ctx_tokens, TEXTSORT_SETTINGS)
    ss = pick_setting_for_ctx(ctx_tokens, STACKSELECT_SETTINGS)
    return [f"textsort_{ts}", f"stackselect_{ss}"]


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


def _encode(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="*", default=None)
    ap.add_argument(
        "--auto", action="store_true", help="Auto-pick datasets based on ctx length."
    )
    ap.add_argument("--model", "-m", type=str, required=True)  # HF id/path
    ap.add_argument("--served_name", type=str, default="")
    ap.add_argument(
        "--save_dir", "-s", type=str, default=str(ROOT / "results_adaleval")
    )
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    ap.add_argument(
        "--dataset_mode", type=str, default="normal", choices=["normal", "less"]
    )

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    ap.add_argument("--rep", type=int, default=int(os.environ.get("RUN_REP", "1")))
    ap.add_argument("--ctx_len", type=int, default=int(os.environ.get("CTX_LEN", "0")))

    ap.add_argument(
        "--tokenizer_id", type=str, default=os.environ.get("TOKENIZER_ID", "")
    )

    ap.add_argument(
        "--dtype", type=str, default=os.environ.get("VLLM_DTYPE", "bfloat16")
    )
    ap.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "1")))
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")),
    )

    ap.add_argument("--force_chat", action="store_true")
    ap.add_argument("--force_completion", action="store_true")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)
    return ap.parse_args()


def build_llm(args) -> LLM:
    return LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=(args.ctx_len if args.ctx_len > 0 else None),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def _chunked(xs: List[Any], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def run_eval(args, llm: Optional[LLM] = None) -> Dict[str, Any]:
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    ctx_len = int(args.ctx_len or 0)
    if (args.auto or not args.data) and ctx_len <= 0:
        raise RuntimeError(
            "Need --ctx_len (or CTX_LEN env) for Ada-LEval auto dataset selection."
        )

    if args.auto or not args.data:
        args.data = auto_datasets_for_ctx(ctx_len)

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

    run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed but --wandb was set.")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", served_name)
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "adaleval-vllm"),
            entity=os.environ.get("WANDB_ENTITY"),
            group=os.environ.get("WANDB_GROUP"),
            name=os.environ.get("WANDB_NAME")
            or f"{safe}__adaleval__ctx{ctx_len}__rep{args.rep}",
            config={
                "model/served_name": served_name,
                "model/hf_id": args.model,
                "tokenizer_id": tokenizer_id,
                "protocol/seed": args.seed,
                "protocol/temperature": args.temperature,
                "protocol/top_p": args.top_p,
                "protocol/max_new_tokens": args.max_new_tokens,
                "protocol/batch_size": args.batch_size,
                "protocol/rep": args.rep,
                "protocol/ctx_len": ctx_len,
                "dataset_mode": args.dataset_mode,
                "datasets": list(args.data or []),
                "use_chat_template": use_chat,
                "engine/dtype": args.dtype,
                "engine/tp": args.tp,
                "engine/gpu_memory_utilization": args.gpu_memory_utilization,
            },
        )

    sp = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
    )

    lat_e2e_all: List[float] = []
    out_tokens_all: List[int] = []
    wall_time_total = 0.0
    total_seen = 0
    accs: Dict[str, float] = {}

    with VramSampler(poll_s=0.1) as vs:
        for dname in args.data or []:
            d, setting = dname.split("_", 1)

            if d == "stackselect":
                dataset = StackSelect(setting=setting, mode=args.dataset_mode)
            elif d == "textsort":
                dataset = TextSort(setting=setting, mode=args.dataset_mode)
            else:
                raise ValueError(f"Unknown dataset prefix in {dname}")

            meta = dataset.get_meta()
            indices = list(meta["index"])
            prompts = [dataset.build_prompt(i) for i in range(len(dataset))]

            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", served_name)
            out_file = (
                save_dir / f"{safe_model}__{dname}__ctx{ctx_len}__rep{args.rep}.pkl"
            )
            res: Dict[Any, Any] = {} if not out_file.exists() else load(out_file)

            to_run: List[Tuple[Any, str]] = []
            for idx, ptxt in zip(indices, prompts):
                if idx in res:
                    if isinstance(res[idx], str):
                        continue
                    if isinstance(res[idx], dict) and "text" in res[idx]:
                        continue
                to_run.append((idx, ptxt))

            for batch in tqdm(list(_chunked(to_run, args.batch_size)), desc=dname):
                batch_indices = [x[0] for x in batch]
                batch_prompts = [
                    maybe_apply_chat_template(tok, x[1], use_chat) for x in batch
                ]

                t0 = time.perf_counter()
                outs = llm.generate(batch_prompts, sp)  # type: ignore[arg-type]
                t1 = time.perf_counter()
                bt = t1 - t0
                wall_time_total += bt

                texts: List[str] = []
                for o in outs:
                    txt = ""
                    if o.outputs:
                        txt = o.outputs[0].text or ""
                    texts.append(txt)

                for idx, txt in zip(batch_indices, texts):
                    try:
                        n_out = len(_encode(tok, txt))
                    except Exception:
                        n_out = 0
                    res[idx] = {
                        "text": txt,
                        "latency_e2e_s": float(bt),
                        "out_tokens": int(n_out),
                    }
                    dump(res, str(out_file))

                    lat_e2e_all.append(float(bt))
                    out_tokens_all.append(int(n_out))
                    total_seen += 1

                    if run is not None and (total_seen % args.log_every == 0):
                        wandb.log(
                            {
                                "progress/seen": total_seen,
                                "eff/latency_e2e_s_mean_sofar": float(
                                    statistics.mean(lat_e2e_all)
                                )
                                if lat_e2e_all
                                else float("nan"),
                                "debug/reprompt_used_sofar": 0,
                            }
                        )

            pred_texts: List[str] = []
            for k in meta["index"]:
                v = res[k]
                pred_texts.append(v if isinstance(v, str) else v.get("text", ""))

            meta["prediction"] = pred_texts

            acc = dataset.evaluate(meta)
            try:
                acc_f = (
                    float(acc)
                    if not isinstance(acc, dict)
                    else float(acc.get("acc", acc.get("accuracy", float("nan"))))
                )
            except Exception:
                acc_f = float("nan")

            accs[dname] = acc_f
            if run is not None:
                wandb.log({f"acc/{dname}": acc_f})

    e2e_mean = float(statistics.mean(lat_e2e_all)) if lat_e2e_all else float("nan")
    e2e_p50 = float(percentile(lat_e2e_all, 0.50))
    e2e_p95 = float(percentile(lat_e2e_all, 0.95))

    total_out = float(sum(out_tokens_all))
    toks_per_s = (total_out / wall_time_total) if wall_time_total > 0 else float("nan")

    peak_vram = float(getattr(vs, "peak_mib_per_gpu_max", float("nan")))
    acc_overall = (
        float(statistics.mean([v for v in accs.values() if v == v]))
        if accs
        else float("nan")
    )

    summary = {
        "acc_overall": acc_overall,
        "acc_by_dataset": accs,
        "e2e_mean_s": e2e_mean,
        "e2e_p50_s": e2e_p50,
        "e2e_p95_s": e2e_p95,
        "ttft_mean_s": float("nan"),
        "ttft_p50_s": float("nan"),
        "ttft_p95_s": float("nan"),
        "tokens_per_s": toks_per_s,
        "peak_vram_mib_per_gpu_max": peak_vram,
        "reprompt_used": 0,
        "save_dir": str(save_dir),
        "datasets": list(args.data or []),
        "wall_time_s": float(wall_time_total),
    }

    print(json.dumps(summary, indent=2))

    if run is not None:
        wandb.log(
            {
                "acc/overall": acc_overall,
                "eff/latency_e2e_s_mean": e2e_mean,
                "eff/latency_e2e_s_p50": e2e_p50,
                "eff/latency_e2e_s_p95": e2e_p95,
                "eff/output_tokens_total": total_out,
                "eff/tokens_per_s": toks_per_s,
                "eff/peak_vram_mib_per_gpu_max": peak_vram,
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
