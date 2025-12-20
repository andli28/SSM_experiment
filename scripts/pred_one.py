#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

ROOT = Path(__file__).resolve().parent.parent

import pred_lb
import pred_leval

from vllm import LLM  # type: ignore


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def build_llm(model_cfg: Dict[str, Any], g: Dict[str, Any], ctx: int) -> LLM:
    hf_id = model_cfg["hf_id"]
    vllm_cfg = g.get("vllm", {})

    dtype = vllm_cfg.get("dtype", "bfloat16")
    tp = int(vllm_cfg.get("tensor_parallel_size", vllm_cfg.get("tp", 1)))
    gpu_mem = float(vllm_cfg.get("gpu_memory_utilization", 0.90))
    
    # Parse extra vLLM args from model config
    extra_args_str = model_cfg.get("vllm_extra_args", "").strip()
    extra_kwargs = {}
    if extra_args_str:
        import shlex
        extra_args = shlex.split(extra_args_str)
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if arg.startswith("--"):
                key = arg[2:].replace("-", "_")
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    extra_kwargs[key] = extra_args[i + 1]
                    i += 2
                else:
                    extra_kwargs[key] = True
                    i += 1
            else:
                i += 1

    return LLM(
        model=hf_id,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=tp,
        max_model_len=int(ctx),
        gpu_memory_utilization=gpu_mem,
        **extra_kwargs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--global_cfg", default=str(ROOT / "configs" / "global.json"))
    args = ap.parse_args()

    g = read_json(Path(args.global_cfg))
    m = read_json(Path(args.model_cfg))

    served = m.get("served", m["hf_id"])
    family = m.get("family", served)

    repeats = int(g.get("repeats", 1))
    seed = int(g.get("seed", 42))

    results_dir = (ROOT / g.get("paths", {}).get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = str(g.get("benchmark", "longbench")).lower()
    batch_size = int(g.get("batch_size", g.get("vllm", {}).get("batch_size", 16)))

    wandb_cfg = g.get("wandb", {})
    wandb_enable = bool(wandb_cfg.get("enable", True))
    if wandb_enable:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "bench-vllm"))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]
        os.environ["WANDB_GROUP"] = family

    t0 = time.perf_counter()
    llm = build_llm(m, g, args.ctx)
    t1 = time.perf_counter()

    summaries: List[Dict[str, Any]] = []

    for rep in range(1, repeats + 1):
        os.environ["SEED"] = str(seed)
        os.environ["RUN_REP"] = str(rep)
        os.environ["CTX_LEN"] = str(args.ctx)
        os.environ["TOKENIZER_ID"] = str(m.get("tokenizer_id", m["hf_id"]))

        if bench in ("longbench", "lb"):
            if wandb_enable:
                os.environ["WANDB_NAME"] = (
                    f"{served}__longbench__ctx{args.ctx}__rep{rep}"
                )

            lb_args = argparse.Namespace(
                save_dir=str(results_dir),
                cache_dir=g.get(
                    "hf_cache_dir",
                    os.environ.get("HF_DATASETS_CACHE", str(ROOT / "hf_datasets")),
                ),
                model=m["hf_id"],
                served_name=served,
                tokenizer_id=m.get("tokenizer_id", m["hf_id"]),
                max_len=int(args.ctx),
                seed=seed,
                rep=rep,
                temperature=float(g.get("longbench", {}).get("temperature", 0.1)),
                top_p=float(g.get("longbench", {}).get("top_p", 1.0)),
                max_new_tokens=int(g.get("longbench", {}).get("max_new_tokens", 128)),
                batch_size=batch_size,
                dtype=g.get("vllm", {}).get("dtype", "bfloat16"),
                tp=int(
                    g.get("vllm", {}).get(
                        "tensor_parallel_size", g.get("vllm", {}).get("tp", 1)
                    )
                ),
                gpu_memory_utilization=float(
                    g.get("vllm", {}).get("gpu_memory_utilization", 0.90)
                ),
                force_chat=bool(g.get("longbench", {}).get("force_chat", False)),
                force_completion=bool(
                    g.get("longbench", {}).get("force_completion", False)
                ),
                wandb=wandb_enable,
                log_every=int(g.get("wandb", {}).get("log_every", 200)),
            )
            summaries.append(
                {"bench": "longbench", "rep": rep, **pred_lb.run_eval(lb_args, llm=llm)}
            )

        elif bench in ("adaleval", "ada-leval", "leval"):
            if wandb_enable:
                os.environ["WANDB_NAME"] = (
                    f"{served}__adaleval__ctx{args.ctx}__rep{rep}"
                )

            adacfg = g.get("adaleval", {})
            datasets = adacfg.get("datasets", "auto")  # "auto" or list

            leval_args = argparse.Namespace(
                data=(None if datasets == "auto" else datasets),
                auto=bool(datasets == "auto"),
                model=m["hf_id"],
                served_name=served,
                save_dir=str(results_dir),
                data_dir=str((ROOT / "data").resolve()),
                dataset_mode=str(adacfg.get("dataset_mode", "normal")),
                temperature=float(adacfg.get("temperature", 0.0)),
                top_p=float(adacfg.get("top_p", 1.0)),
                max_new_tokens=int(adacfg.get("max_new_tokens", 16)),
                batch_size=batch_size,
                seed=seed,
                rep=rep,
                ctx_len=int(args.ctx),
                tokenizer_id=m.get("tokenizer_id", m["hf_id"]),
                dtype=g.get("vllm", {}).get("dtype", "bfloat16"),
                tp=int(
                    g.get("vllm", {}).get(
                        "tensor_parallel_size", g.get("vllm", {}).get("tp", 1)
                    )
                ),
                gpu_memory_utilization=float(
                    g.get("vllm", {}).get("gpu_memory_utilization", 0.90)
                ),
                force_chat=bool(adacfg.get("force_chat", False)),
                force_completion=bool(adacfg.get("force_completion", False)),
                wandb=wandb_enable,
                log_every=int(g.get("wandb", {}).get("log_every", 200)),
            )
            summaries.append(
                {
                    "bench": "adaleval",
                    "rep": rep,
                    **pred_leval.run_eval(leval_args, llm=llm),
                }
            )

        else:
            raise ValueError(
                f"Unknown benchmark '{bench}' (use 'longbench' or 'adaleval')."
            )

    out = {
        "model": m.get("name", served),
        "hf_id": m["hf_id"],
        "served": served,
        "ctx": int(args.ctx),
        "repeats": repeats,
        "batch_size": batch_size,
        "load_time_s": float(t1 - t0),
        "summaries": summaries,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
