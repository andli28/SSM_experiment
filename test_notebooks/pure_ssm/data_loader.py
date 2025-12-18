# data_loader.py
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Path to the repo root (where this file lives)
REPO_ROOT = Path(__file__).resolve().parent

# LongBench v2 prompt-set directory
LBV2_PROMPT_ROOT = REPO_ROOT / "data" / "prompt_sets" / "longbench_v2"

# Map between ctx lengths and the *_8k/*.jsonl tags
CTX_TAG_BY_LEN = {
    8192: "8k",
    16384: "16k",
    32768: "32k",
}
LEN_BY_CTX_TAG = {v: k for k, v in CTX_TAG_BY_LEN.items()}


def _load_lb_file(path: Path) -> Tuple[List[str], List[str]]:
    """
    Internal helper: load a LongBench v2 jsonl file into (prompts, labels).

    Assumes each JSON line has at least:
      - "prompt": the full text prompt (context + question + choices)
      - "target": the gold MC option ("A"/"B"/"C"/"D")
    """
    prompts: List[str] = []
    labels: List[str] = []

    if not path.exists():
        raise FileNotFoundError(f"LongBench v2 file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = row["prompt"]      # adjust if your schema differs
            gold = row["target"]        # "A"/"B"/"C"/"D"
            prompts.append(prompt)
            labels.append(gold)

    return prompts, labels


def load_longbench_v2(ctx_tag: str = "8k") -> Tuple[List[str], List[str]]:
    """
    Public helper: load LongBench v2 prompts for a given context tag
    ("8k", "16k", or "32k").

    Returns (prompts, labels).
    """
    if ctx_tag not in LEN_BY_CTX_TAG:
        raise ValueError(f"Unknown ctx_tag={ctx_tag!r}, "
                         f"expected one of {list(LEN_BY_CTX_TAG.keys())}")

    path = LBV2_PROMPT_ROOT / f"lbv2_{ctx_tag}.jsonl"
    return _load_lb_file(path)


# ---------------------------------------------------------------------
# Pre-load any LongBench v2 prompt sets that actually exist on disk.
# These dicts are keyed by *context length in tokens* (8192, 16384, 32768).
# ---------------------------------------------------------------------

LB_V2_PROMPTS_BY_CTX: Dict[int, List[str]] = {}
LB_V2_LABELS_BY_CTX: Dict[int, List[str]] = {}

for ctx_len, ctx_tag in CTX_TAG_BY_LEN.items():
    path = LBV2_PROMPT_ROOT / f"lbv2_{ctx_tag}.jsonl"
    if path.exists():
        prompts, labels = _load_lb_file(path)
        LB_V2_PROMPTS_BY_CTX[ctx_len] = prompts
        LB_V2_LABELS_BY_CTX[ctx_len] = labels

# ---------------------------------------------------------------------
# Backwards-compatible 8k objects (used in older Phase‑4 code)
# ---------------------------------------------------------------------

lb_prompts_8k = LB_V2_PROMPTS_BY_CTX.get(8192, [])
longbench_v2_labels = LB_V2_LABELS_BY_CTX.get(8192, [])

datasets_8k = {
    "longbench_v2": lb_prompts_8k,
}
