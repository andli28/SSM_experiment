from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import json

from transformers import AutoTokenizer


# ---------- paths & IO ----------

def get_ada_root(prompt_root: Path) -> Path:
    """Directory under PROMPT_ROOT where all Ada-LEval prompt sets live."""
    root = Path(prompt_root) / "ada_leval"
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


# ---------- raw StackSelect loader ----------

ADA_REPO_PATH = Path("/content/Ada-LEval")  # where the repo lives


def get_stackselect_path(setting: str = "8k") -> Path:
    """
    BestAnswer (StackSelect) files live at:
        /content/Ada-LEval/data/stackselect_8k.json
        /content/Ada-LEval/data/stackselect_16k.json
        /content/Ada-LEval/data/stackselect_32k.json
    """
    path = ADA_REPO_PATH / "data" / f"stackselect_{setting}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    return path


def load_bestanswer_raw(setting: str = "8k") -> List[Dict]:
    """
    Load raw BestAnswer (StackSelect) samples for a given setting.
    The file is a JSON list of dicts.
    """
    path = get_stackselect_path(setting)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[Ada BestAnswer] Loaded {len(raw)} samples from {path}")
    return raw


# ---------- tokenizer + truncation-based prompt builder ----------

def get_ada_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def build_truncated_bestanswer_prompt_for_ctx(
    sample: Dict,
    tokenizer,
    target_ctx_len: int,
    max_new_tokens: int,
    safety_margin: int = 32,
) -> Tuple[str, int] | None:
    """
    Build a *single* prompt that fits inside target_ctx_len (input side),
    by truncating the combined answer texts if needed.

    Fields in sample (from your printout):
      - question_id: int
      - question: str
      - all_answers: List[str]   # candidate answers
      - answer: str              # e.g. "A7"
      - tags: List[str]
    """
    question      = sample["question"]
    all_answers   = sample["all_answers"]  # list of strings
    answer_labels = [f"A{i+1}" for i in range(len(all_answers))]

    # Header: instructions + question
    header = (
        "You are a helpful programming assistant.\n\n"
        "You are given a Stack Overflow question and several candidate answers.\n"
        "Each answer is labeled with an ID like A1, A2, ..., An.\n"
        "Your job is to choose which candidate answer best answers the question.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate answers:\n"
    )

    # Context body: the (potentially long) list of candidate answers
    ctx_lines = []
    for aid, ans_text in zip(answer_labels, all_answers):
        ctx_lines.append(f"{aid}:\n{ans_text}\n")
    ctx = "\n".join(ctx_lines)

    # Tail: final instruction about output format
    tail = (
        "\n\nThink briefly about which candidate is the best answer.\n"
        "Then respond with ONLY the ID of the best answer, for example: A3.\n"
    )

    # Tokenize pieces separately so we can truncate ctx if needed
    header_ids = tokenizer.encode(header, add_special_tokens=False)
    tail_ids   = tokenizer.encode(tail, add_special_tokens=False)
    ctx_ids    = tokenizer.encode(ctx, add_special_tokens=False)

    hard_max_input = target_ctx_len - max_new_tokens - safety_margin
    usable_for_ctx = hard_max_input - len(header_ids) - len(tail_ids)

    if usable_for_ctx <= 0:
        # Non-context parts already too long
        return None

    if len(ctx_ids) > usable_for_ctx:
        ctx_ids = ctx_ids[:usable_for_ctx]

    truncated_ctx = tokenizer.decode(ctx_ids)
    prompt = header + truncated_ctx + tail

    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n_tokens > hard_max_input:
        # Very defensive; should be rare
        print(
            f"[Ada BestAnswer WARN] prompt still too long "
            f"(n_tokens={n_tokens}, limit={hard_max_input}). Skipping."
        )
        return None

    return prompt, n_tokens


# ---------- public builders / loaders ----------

def build_ada_bestanswer_prompt_sets(
    prompt_root: Path,
    tokenizer_model_id: str,
    pure_ssm_contexts: List[int],
    max_new_tokens: int,
    setting: str = "8k",           # which stackselect_* file we use
    max_examples_per_ctx: int = 200,
) -> None:
    """
    For each context length in pure_ssm_contexts, build a JSONL file:
        <prompt_root>/ada_leval/ada_bestanswer_8k.jsonl
        <prompt_root>/ada_leval/ada_bestanswer_16k.jsonl
        <prompt_root>/ada_leval/ada_bestanswer_32k.jsonl

    Each record:
      {
        "prompt": <string>,
        "target": <correct ID string, e.g. "A7">,
        "dataset": "ada_leval.bestanswer",
        "split": <setting>,
        "example_id": <question_id>,
        "meta": { ... }
      }
    """
    prompt_root = Path(prompt_root)
    ada_root    = get_ada_root(prompt_root)

    raw = load_bestanswer_raw(setting=setting)
    tok = get_ada_tokenizer(tokenizer_model_id)

    print("Building Ada-LEval BestAnswer prompt sets.")
    print("Tokenizer:", tokenizer_model_id)
    print("Contexts:", pure_ssm_contexts, "max_new_tokens:", max_new_tokens)

    for ctx_len in pure_ssm_contexts:
        tag = f"{ctx_len // 1024}k"
        records: List[Dict] = []

        for i, sample in enumerate(raw):
            out = build_truncated_bestanswer_prompt_for_ctx(
                sample=sample,
                tokenizer=tok,
                target_ctx_len=ctx_len,
                max_new_tokens=max_new_tokens,
            )
            if out is None:
                continue

            prompt, n_tokens = out

            gold_id = sample["answer"]  # e.g. "A7"

            rec = {
                "prompt": prompt,
                "target": gold_id,
                "dataset": "ada_leval.bestanswer",
                "split": setting,
                "example_id": sample.get("question_id", i),
                "meta": {
                    "setting": setting,
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
                f"[Ada BestAnswer ctx={ctx_len}] collected={len(records)}, "
                f"min_len={min(lens)}, max_len={max(lens)}, "
                f"mean_len={sum(lens)/len(lens):.1f}"
            )
        else:
            print(f"[Ada BestAnswer ctx={ctx_len}] WARNING: collected 0 records.")

        out_path = ada_root / f"ada_bestanswer_{tag}.jsonl"
        save_prompt_records(records, out_path)


def load_ada_bestanswer_prompts_for_tag(
    prompt_root: Path,
    tag: str,  # "8k", "16k", "32k"
) -> Tuple[List[Dict], List[str]]:
    """
    Load the frozen Ada-LEval prompt set for a given tag.
    """
    prompt_root = Path(prompt_root)
    ada_root    = get_ada_root(prompt_root)
    path        = ada_root / f"ada_bestanswer_{tag}.jsonl"

    records = load_prompt_records(path)
    prompts = [r["prompt"] for r in records]
    return records, prompts
