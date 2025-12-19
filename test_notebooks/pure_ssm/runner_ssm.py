# runner_ssm.py
"""
Runtime helpers for the Pure SSM track.

- RunConfig: describes a single (model, context_len) run.
- load_llm: creates a vLLM LLM from a RunConfig.
- make_run_configs_for_pure_ssm: builds the (model, ctx) grid.
- smoke_test_pure_ssm: quick sanity test for Mamba at 8k.
- generate_batch / profile_run / log_result: Phase 2 helpers for
  inference, profiling, and logging.

Extras:
- LB_ALEVAL_SAMPLING: safer SamplingParams for LongBench / Ada-style prompts.
- score_mc_options / choose_mc_option: logprob-based MC scoring.
- sampling_for_dataset: dataset-aware SamplingParams helper.
"""

import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Sequence

import numpy as np
import torch
from vllm import LLM, SamplingParams

from config.pure_ssm_config import PURE_SSM_MODELS, PURE_SSM_CONTEXTS, DECODE_CONFIG

PrecisionType = Literal["float16", "bfloat16"]


# ------------------- RunConfig + seeding ------------------- #


@dataclass
class RunConfig:
    """Configuration for a single Pure SSM run."""
    model_key: str          # "mamba-2.8b" or "mamba-codestral-7b"
    context_len: int        # 8192 / 16384 / 32768

    # Decoding / runtime knobs
    max_new_tokens: int = DECODE_CONFIG["max_new_tokens"]
    precision: PrecisionType = "bfloat16"
    gpu_mem_util: float = 0.90
    seed: int = 42

    @property
    def model_id(self) -> str:
        return PURE_SSM_MODELS[self.model_key]["hf_id"]

    @property
    def revision(self) -> str:
        return PURE_SSM_MODELS[self.model_key]["revision"]

    @property
    def max_context_for_model(self) -> int:
        return PURE_SSM_MODELS[self.model_key]["max_context"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------- LLM loader ------------------- #


def load_llm(cfg: RunConfig) -> LLM:
    """
    Instantiate a vLLM LLM for the given Pure SSM configuration.
    Uses trust_remote_code=True for Mamba/Mamba-2 repos.
    """
    set_global_seed(cfg.seed)

    llm = LLM(
        model=cfg.model_id,
        revision=cfg.revision,
        tokenizer_revision=cfg.revision,
        trust_remote_code=True,
        dtype=cfg.precision,
        tensor_parallel_size=1,
        max_model_len=cfg.context_len,
        gpu_memory_utilization=cfg.gpu_mem_util,
        enforce_eager=True,   # matches your env smoke test
    )
    return llm


# ------------------- Shared sampling params ------------------- #


DEFAULT_SAMPLING = SamplingParams(
    temperature=DECODE_CONFIG["temperature"],
    top_p=DECODE_CONFIG["top_p"],
    max_tokens=DECODE_CONFIG["max_new_tokens"],
)

# Used for "normal" LongBench / Ada-style generation (not MC scoring)
LB_ALEVAL_SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=32,
    stop=[
        "\n\n",
        "\nQuestion:",
        "\nContext:",
        "Choices:",
        "Answer with a single letter:",
    ],
)



# ---- Optional: logprob-based MC scoring for LongBench / Ada-LEval ---- #


def score_mc_options(
    llm: LLM,
    prompt: str,
    options: Sequence[str],   # e.g. ["A", "B", "C", "D"]
    *,
    # what you put before the answer letter, e.g. " " or "\nAnswer: "
    answer_prefix: str = " ",
    temperature: float = 0.0,
    top_p: float = 1.0,
    prompt_logprobs_k: int = 16,
) -> Dict[str, float]:
    """
    Compute log p(answer_prefix + option | prompt) for each option using
    vLLM's prompt_logprobs.

    This version only uses vLLM's own tokenization, so it works even for
    models like Codestral whose tokenizer differs from HF's.
    """

    # One shared SamplingParams:
    # - deterministic (temperature=0, top_p=1)
    # - we don't care about generated tokens, only prompt_logprobs
    sp = SamplingParams(
        max_tokens=1,          # must be >=1, but we ignore the generated token
        temperature=temperature,
        top_p=top_p,
        prompt_logprobs=prompt_logprobs_k,
        logprobs=0,
    )

    # We send one request for the bare prompt and one per option
    prompts = [prompt] + [prompt + answer_prefix + opt for opt in options]
    results = llm.generate(prompts, sp)

    base_res = results[0]
    base_len = len(base_res.prompt_token_ids)

    # We'll only use the tokenizer to map token IDs to text if needed
    tokenizer = llm.get_tokenizer()

    scores: Dict[str, float] = {}

    for opt, res in zip(options, results[1:]):
        full_ids = res.prompt_token_ids
        full_lps = res.prompt_logprobs
        full_len = len(full_ids)

        # Extra tokens added by "answer_prefix + opt"
        extra = full_len - base_len
        if extra <= 0:
            # Something odd (aggressive truncation etc.) -> treat as very unlikely
            scores[opt] = float("-inf")
            continue

        # The *last* `extra` prompt tokens are the answer tokens
        start = full_len - extra  # indices [start, full_len) are the answer tokens
        logp = 0.0

        for idx in range(start, full_len):
            tok_id = full_ids[idx]
            lp_dict = full_lps[idx]

            if not lp_dict:
                # No logprobs returned for this position -> big penalty
                logp += -1e9
                continue

            # Two possible shapes:
            # - keys are token IDs (int)
            # - keys are token strings (str) after detokenization
            entry = lp_dict.get(tok_id)
            if entry is None:
                tok_str = tokenizer.decode([tok_id])
                entry = lp_dict.get(tok_str)

            if entry is None:
                lp = -1e9
            else:
                # In newer vLLM, entry is a LogprobResult with `.logprob`
                lp = float(getattr(entry, "logprob", entry))

            logp += lp

        scores[opt] = logp

    return scores


def choose_mc_option(
    llm: LLM,
    prompt: str,
    options: Sequence[str] = ("A", "B", "C", "D"),
    *,
    answer_prefix: str = " ",
) -> str:
    """
    Return the option (e.g. 'A', 'B', 'C', 'D') with highest log-prob.
    """
    scores = score_mc_options(
        llm,
        prompt,
        options=options,
        answer_prefix=answer_prefix,
    )
    # argmax over options; ties broken by first-in-order
    best_opt = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_opt



# --- Dataset-aware sampling helpers (for generative runs) --- #

# These are for your *generation* runs (run_and_log),
# not for logprob-based MC scoring.

LONGBENCH_STOP = [
    # Markers that appear after an answer in your prompts.
    # Adjust these based on your actual LongBench wrapper.
    "\n\nQuestion:",
    "\n\nContext:",
    "\n\n---",
]

ADA_LEVAL_STOP = [
    # Adjust based on how your Ada-LEval prompts are formatted.
    "\n\n[END OF PROBLEM]",
    "\n\n--- NEW SAMPLE ---",
]


def make_sampling_params(
    *,
    stop=None,
    temperature=None,
    top_p=None,
    max_tokens=None,
) -> SamplingParams:
    """Utility to build SamplingParams with project defaults + optional overrides."""
    return SamplingParams(
        temperature=DECODE_CONFIG["temperature"] if temperature is None else temperature,
        top_p=DECODE_CONFIG["top_p"] if top_p is None else top_p,
        max_tokens=DECODE_CONFIG["max_new_tokens"] if max_tokens is None else max_tokens,
        stop=stop,
    )


def sampling_for_dataset(dataset_name: str) -> SamplingParams:
    """
    Build SamplingParams for a given dataset.

    - LongBench / Ada-LEval: greedy decoding + stop sequences
      to prevent long, repetitive rambling.
    - PG-19 (and anything else): fall back to DEFAULT_SAMPLING.
    """
    if dataset_name == "longbench_v2":
        return make_sampling_params(
            stop=LONGBENCH_STOP,
            temperature=0.0,
            top_p=1.0,
        )
    if dataset_name in {"ada_bestanswer", "ada_leval"}:
        return make_sampling_params(
            stop=ADA_LEVAL_STOP,
            temperature=0.0,
            top_p=1.0,
        )

    # Default (e.g., PG-19)
    return DEFAULT_SAMPLING


# ------------------- Generation + profiling (Phase 2) ------------------- #


def generate_batch(
    llm: LLM,
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
    batch_size: int = 1,
) -> List[str]:
    """
    Run vLLM.generate on a list of prompts and return plain-text outputs.

    For fairness in the project, keep batch_size=1 in main experiments.
    """
    if sampling_params is None:
        sampling_params = DEFAULT_SAMPLING

    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i: i + batch_size]
        results = llm.generate(chunk, sampling_params)
        for r in results:
            if not r.outputs:
                outputs.append("")
            else:
                outputs.append(r.outputs[0].text)
    return outputs


def current_vram_gb() -> float:
    """
    Query the current GPU memory usage (in GB) using nvidia-smi.

    Returns 0.0 if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        mb = int(out.decode().strip().split("\n")[0])
        return mb / 1024.0
    except Exception:
        return 0.0


def profile_run(
    llm: LLM,
    prompts: List[str],
    cfg: RunConfig,
    tag: str,
    sampling_params: Optional[SamplingParams] = None,
    batch_size: int = 1,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Run generation and collect efficiency stats.

    Returns:
        stats: dict with timing, tokens, VRAM, and model/context metadata
        outputs: list of completions (one per prompt)
    """
    if sampling_params is None:
        sampling_params = DEFAULT_SAMPLING

    # Clean CUDA state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # VRAM before generation
    start_vram = current_vram_gb()
    t0 = time.time()

    # Actual generation
    outputs = generate_batch(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    # Time + VRAM after
    t1 = time.time()
    end_vram = current_vram_gb()
    peak_alloc_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_alloc_gb = peak_alloc_bytes / (1024 ** 3) if peak_alloc_bytes else 0.0

    # Token accounting
    tokenizer = llm.get_tokenizer()
    input_tokens = sum(len(tokenizer.encode(p)) for p in prompts)
    output_tokens = sum(len(tokenizer.encode(o)) for o in outputs)
    total_tokens = input_tokens + output_tokens
    total_time_s = t1 - t0
    tokens_per_s = total_tokens / total_time_s if total_time_s > 0 else 0.0

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    stats: Dict[str, Any] = {
        "tag": tag,
        "model_key": cfg.model_key,
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "context_len": cfg.context_len,
        "max_new_tokens": cfg.max_new_tokens,
        "precision": cfg.precision,
        "gpu_mem_util": cfg.gpu_mem_util,
        "seed": cfg.seed,
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "tokens_per_s": tokens_per_s,
        "start_vram_gb": start_vram,
        "peak_vram_gb": max(start_vram, peak_alloc_gb),
        "end_vram_gb": end_vram,
        "peak_allocated_gb": peak_alloc_gb,
        "device_name": device_name,
    }

    return stats, outputs


def log_result(
    stats: Dict[str, Any],
    outputs: List[str],
    prompts: List[str],
    log_name: str,
    log_dir_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Append stats + (prompt, completion) pairs to a JSONL file.

    Args:
        stats: run-level statistics dictionary (from profile_run).
        outputs: list of completions.
        prompts: list of prompts (same length as outputs).
        log_name: base name for the JSONL file (without extension).
        log_dir_root: if provided, logs under <root>/pure_ssm_logs;
                      otherwise under CWD/pure_ssm_logs.

    Returns:
        Path to the JSONL log file.
    """
    if log_dir_root is None:
        log_dir_root = Path.cwd()
    else:
        log_dir_root = Path(log_dir_root)

    log_dir = log_dir_root / "pure_ssm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / f"{log_name}.jsonl"

    with path.open("a", encoding="utf-8") as f:
        for i, (p, o) in enumerate(zip(prompts, outputs)):
            row = dict(stats)
            row.update(
                {
                    "sample_idx": i,
                    "prompt": p,
                    "completion": o,
                }
            )
            f.write(json.dumps(row) + "\n")

    print(f"Logged {len(outputs)} samples to {path}")
    return path


def run_and_log(
    cfg: RunConfig,
    prompts: List[str],
    tag: str,
    log_name: str,
    log_dir_root: Optional[Union[str, Path]] = None,
    sampling_params: Optional[SamplingParams] = None,
    batch_size: int = 1,
) -> Tuple[Dict[str, Any], List[str], Path]:
    """
    High-level helper:
      - loads model
      - profiles generation
      - logs results
      - returns (stats, outputs, log_path)
    """
    llm = load_llm(cfg)
    stats, outputs = profile_run(
        llm=llm,
        prompts=prompts,
        cfg=cfg,
        tag=tag,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    path = log_result(
        stats=stats,
        outputs=outputs,
        prompts=prompts,
        log_name=log_name,
        log_dir_root=log_dir_root,
    )
    return stats, outputs, path


# ------------------- Config grid helpers ------------------- #


def make_run_configs_for_pure_ssm() -> List[RunConfig]:
    """
    Create one RunConfig for each (model, context_len) pair,
    skipping any context_len > model.max_context.
    """
    cfgs: List[RunConfig] = []
    for model_key, m_cfg in PURE_SSM_MODELS.items():
        max_ctx_model = m_cfg["max_context"]
        for ctx in PURE_SSM_CONTEXTS:
            if ctx <= max_ctx_model:
                cfgs.append(RunConfig(model_key=model_key, context_len=ctx))
    return cfgs


# ------------------- Smoke test ------------------- #


def smoke_test_pure_ssm() -> None:
    """
    Load one Pure SSM model at 8k and generate a short reply.
    This proves that vLLM + Mamba work end-to-end on your env.
    """
    cfgs = make_run_configs_for_pure_ssm()
    # pick the first config at 8k
    cfg = next(c for c in cfgs if c.context_len == 8192)

    print(f"[SMOKE SSM] Loading {cfg.model_key} ({cfg.model_id}) @ ctx={cfg.context_len}")
    llm = load_llm(cfg)

    prompt = "You are a long-context Mamba-style model. Say hello in one short sentence."
    outputs = llm.generate([prompt], DEFAULT_SAMPLING)

    print("Output:", outputs[0].outputs[0].text)
    print("✅ Pure SSM smoke test finished.")


if __name__ == "__main__":
    # Allow `python runner_ssm.py` to run the smoke test directly
    smoke_test_pure_ssm()
