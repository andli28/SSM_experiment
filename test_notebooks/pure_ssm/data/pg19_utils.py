from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import json

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------- paths & IO ----------

def get_pg19_root(prompt_root: Path) -> Path:
    """Directory under PROMPT_ROOT where all PG-19 chunks live."""
    root = Path(prompt_root) / "pg19"
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

def load_pg19(split: str = "test"):
    """
    Load PG-19 from the small mirror `emozilla/pg19-test`.
    This is much lighter than `deepmind/pg19` but has the same 100 test books.
    """
    ds = load_dataset("emozilla/pg19-test", split=split)
    print(ds)
    return ds


def get_pg19_tokenizer(model_id: str):
    """Tokenizer used to measure/construct chunks."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok


# ---------- truncation to a fixed context length ----------

def build_pg19_chunk_for_ctx(
    sample: Dict,
    tokenizer,
    target_ctx_len: int,
    max_new_tokens: int,
    safety_margin: int = 32,
    min_fraction: float = 0.75,
) -> Tuple[str, int, int] | None:
    """
    Build a text chunk from a PG-19 book that fits into target_ctx_len (input side)
    by truncating if needed.

    Returns (chunk_text, chunk_tokens, full_tokens) or None.
    """
    full_text = sample["text"]

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    full_len = len(full_ids)

    hard_max_input = target_ctx_len - max_new_tokens - safety_margin

    # Skip books that are too short for this context length
    if full_len < int(min_fraction * hard_max_input):
        return None

    chunk_ids = full_ids[:hard_max_input]
    chunk_text = tokenizer.decode(chunk_ids)
    chunk_len = len(chunk_ids)

    # Extra safety: re-encode and verify length
    n_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
    if n_tokens > hard_max_input:
        print(
            f"[PG19 WARN] chunk still too long after truncation "
            f"(n_tokens={n_tokens}, limit={hard_max_input}). Skipping."
        )
        return None

    return chunk_text, chunk_len, full_len


# ---------- builders / loaders ----------

def build_pg19_prompt_sets(
    prompt_root: Path,
    tokenizer_model_id: str,
    pure_ssm_contexts: List[int],
    max_new_tokens: int,
    split: str = "test",
    max_examples_per_ctx: int = 50,
    min_fraction: float = 0.75,
) -> None:
    """
    For each context length in pure_ssm_contexts, build a JSONL file:
        <prompt_root>/pg19/pg19_8k.jsonl
        <prompt_root>/pg19_16k.jsonl
        <prompt_root>/pg19_32k.jsonl
    """
    prompt_root = Path(prompt_root)
    pg19_root   = get_pg19_root(prompt_root)

    ds  = load_pg19(split=split)
    tok = get_pg19_tokenizer(tokenizer_model_id)

    print("Building PG-19 prompt sets with tokenizer:", tokenizer_model_id)
    print("Contexts:", pure_ssm_contexts, "max_new_tokens:", max_new_tokens)

    for ctx_len in pure_ssm_contexts:
        tag = f"{ctx_len // 1024}k"
        records: List[Dict] = []

        for i, sample in enumerate(ds):
            out = build_pg19_chunk_for_ctx(
                sample=sample,
                tokenizer=tok,
                target_ctx_len=ctx_len,
                max_new_tokens=max_new_tokens,
                safety_margin=32,
                min_fraction=min_fraction,
            )
            if out is None:
                continue

            chunk_text, chunk_len, full_len = out

            rec = {
                "prompt": chunk_text,
                "target": None,
                "dataset": "pg19",
                "split": split,
                "example_id": sample.get("short_book_title", f"book_{i}"),
                "meta": {
                    "short_book_title": sample.get("short_book_title"),
                    "publication_date": sample.get("publication_date"),
                    "url": sample.get("url"),
                    "prompt_tokens": chunk_len,
                    "orig_tokens": full_len,
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
                f"[PG19 ctx={ctx_len}] collected={len(records)}, "
                f"min_len={min(lens)}, max_len={max(lens)}, "
                f"mean_len={sum(lens)/len(lens):.1f}"
            )
        else:
            print(f"[PG19 ctx={ctx_len}] WARNING: collected 0 records.")

        out_path = pg19_root / f"pg19_{tag}.jsonl"
        save_prompt_records(records, out_path)


def load_pg19_prompts_for_tag(
    prompt_root: Path,
    tag: str,  # "8k", "16k", "32k"
) -> Tuple[List[Dict], List[str]]:
    """
    Load frozen PG-19 chunks for a given tag.

    Returns (records, prompts).
    """
    prompt_root = Path(prompt_root)
    pg19_root   = get_pg19_root(prompt_root)
    path        = pg19_root / f"pg19_{tag}.jsonl"

    records = load_prompt_records(path)
    prompts = [r["prompt"] for r in records]
    return records, prompts
