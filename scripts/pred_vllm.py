# pred_vllm.py
import os, json, time, re, argparse, threading, statistics
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import wandb


ROOT = Path(__file__).resolve().parent.parent   # repo root
PROMPTS_DIR = ROOT / "scripts" / "prompts"

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

template_rag = _read(PROMPTS_DIR / "0shot_rag.txt")
template_no_context = _read(PROMPTS_DIR / "0shot_no_context.txt")
template_0shot = _read(PROMPTS_DIR / "0shot.txt")
template_0shot_cot = _read(PROMPTS_DIR / "0shot_cot.txt")
template_0shot_cot_ans = _read(PROMPTS_DIR / "0shot_cot_ans.txt")

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token-abc123")
TOKENIZER_ID_ENV = os.environ.get("TOKENIZER_ID")  # set by run_one.sh

# More robust answer extraction
ANSWER_PATTERNS = [
    re.compile(r"(?:final\s*answer|answer)\s*[:\-]\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-D])\)?\s*$", re.IGNORECASE),
    re.compile(r"\(([A-D])\)"),
]

# ----------------------------
# NVML VRAM sampler (peak MiB)
# ----------------------------
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class VramSampler:
    def __init__(self, poll_s: float = 0.1):
        self.poll_s = poll_s
        self.peak_mib_per_gpu_max = 0.0
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._handles = []

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
                mx = max(mx, info.used / (1024 ** 2))
            self.peak_mib_per_gpu_max = max(self.peak_mib_per_gpu_max, mx)
            time.sleep(self.poll_s)


# ----------------------------
# Tokenizer helpers (HF + tiktoken)
# ----------------------------
def _is_tiktoken(tok) -> bool:
    return isinstance(tok, tiktoken.Encoding)

def _encode(tok, text: str) -> List[int]:
    if _is_tiktoken(tok):
        return tok.encode(text, disallowed_special=())
    return tok.encode(text, add_special_tokens=False)

def _decode(tok, ids: List[int]) -> str:
    if _is_tiktoken(tok):
        return tok.decode(ids)
    return tok.decode(ids, skip_special_tokens=True)

def truncate_prompt(prompt: str, tok, max_len: int) -> str:
    if max_len <= 0:
        return prompt
    ids = _encode(tok, prompt)
    if len(ids) > max_len:
        half = max_len // 2
        ids = ids[:half] + ids[-half:]
        prompt = _decode(tok, ids)
    return prompt


# ----------------------------
# Answer extraction
# ----------------------------
def extract_answer(response: str) -> Optional[str]:
    if not response:
        return None
    text = response.replace("*", "").strip()
    tail = text[-300:]
    for pat in ANSWER_PATTERNS:
        m = pat.search(tail)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", tail)
    return m.group(1).upper() if m else None


# ----------------------------
# Misc helpers
# ----------------------------
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


def build_prompt(item: Dict[str, Any], args) -> Tuple[str, str]:
    context = item["context"]
    if args.rag > 0:
        template = template_rag
        retrieved = item.get("retrieved_context", [])[: args.rag]
        retrieved = sorted(retrieved, key=lambda x: x.get("c_idx", 0))
        context = "\n\n".join(
            [f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)]
        )
    elif args.no_context:
        template = template_no_context
    elif args.cot:
        template = template_0shot_cot
    else:
        template = template_0shot

    prompt = (
        template.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )
    return prompt, context


def stream_chat_completion(
    *,
    client: OpenAI,
    served_model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> Tuple[str, float, float]:
    """
    Returns (text, ttft_seconds, e2e_seconds)
    """
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=served_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
        stream=True,
    )

    t_first = None
    parts: List[str] = []

    for chunk in stream:
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None) or ""
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            parts.append(piece)

    t_end = time.perf_counter()
    text = "".join(parts)
    ttft = (t_first - t0) if t_first is not None else (t_end - t0)
    e2e = t_end - t0
    return text, ttft, e2e


# ----------------------------
# Main eval loop
# ----------------------------
def run_eval(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model_key = args.model
    served_model_name = args.model  # MUST match vLLM --served-model-name
    tokenizer_id = (args.tokenizer_id or TOKENIZER_ID_ENV or model_key)

    # Tokenizer: use tiktoken only if you truly evaluate OpenAI models.
    # For vLLM-served HF models, HF tokenizer is preferred.
    if ("gpt" in model_key or "o1" in model_key) and not tokenizer_id:
        tok = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    client = OpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)

    # Output file: per (model, ctx, rep, flags)
    ctx = args.max_len
    rep = args.rep
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_key)
    suffix = ""
    if args.rag > 0: suffix += f"_rag{args.rag}"
    if args.no_context: suffix += "_noctx"
    if args.cot: suffix += "_cot"

    out_file = os.path.join(args.save_dir, f"{safe}__ctx{ctx}__rep{rep}{suffix}.jsonl")

    dataset = load_dataset(
        "THUDM/LongBench-v2",
        split="train",
        cache_dir=args.cache_dir,
        download_mode="reuse_dataset_if_exists",
    )

    data_all = [
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
            "retrieved_context": item.get("retrieved_context", []),
        }
        for item in dataset
    ]

    # Resume cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                try:
                    has_data[json.loads(line)["_id"]] = 1
                except Exception:
                    pass

    data = [x for x in data_all if x["_id"] not in has_data]
    fout = open(out_file, "a", encoding="utf-8")

    # W&B init
    run = None
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "longbench-vllm"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME"),
            group=os.environ.get("WANDB_GROUP"),
            config={
                "model/served_name": served_model_name,
                "tokenizer_id": tokenizer_id,
                "vllm/url": VLLM_URL,
                "protocol/seed": args.seed,
                "protocol/temperature": 0.0,
                "protocol/top_p": 1.0,
                "cot": bool(args.cot),
                "no_context": bool(args.no_context),
                "rag_k": int(args.rag),
                "ctx_len": args.max_len,
                "rep": args.rep,
            },
        )

    # metrics accumulators
    lat_e2e: List[float] = []
    lat_ttft: List[float] = []
    out_tokens: List[int] = []
    reprompt_used = 0

    # accuracy accumulators
    counts = {"easy": 0, "hard": 0, "short": 0, "medium": 0, "long": 0, "total": 0}
    correct = {"easy": 0, "hard": 0, "short": 0, "medium": 0, "long": 0, "total": 0}

    with VramSampler(poll_s=0.1) as vs:
        for item in tqdm(data):
            prompt, context = build_prompt(item, args)

            max_new = 1024 if args.cot else 128
            prompt_budget = max(256, args.max_len - max_new - 64)
            prompt = truncate_prompt(prompt, tok, prompt_budget)

            # Encourage parsable answer without forcing CoT:
            prompt += "\n\nYou MUST end with: Final answer: X (X is A, B, C, or D)."

            resp, ttft_s, e2e_s = stream_chat_completion(
                client=client,
                served_model_name=served_model_name,
                prompt=prompt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new,
                seed=args.seed,
            )
            if not resp:
                continue

            # Optional explicit CoT mode: ask for reasoning then answer extraction
            if args.cot:
                response_cot = resp.strip()
                item["response_cot"] = response_cot

                prompt2 = (
                    template_0shot_cot_ans.replace("$DOC$", context.strip())
                    .replace("$Q$", item["question"].strip())
                    .replace("$C_A$", item["choice_A"].strip())
                    .replace("$C_B$", item["choice_B"].strip())
                    .replace("$C_C$", item["choice_C"].strip())
                    .replace("$C_D$", item["choice_D"].strip())
                    .replace("$COT$", response_cot)
                )
                prompt2 = truncate_prompt(prompt2, tok, prompt_budget)
                prompt2 += "\n\nFinal answer:"

                resp2, ttft2, e2e2 = stream_chat_completion(
                    client=client,
                    served_model_name=served_model_name,
                    prompt=prompt2,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=16,
                    seed=args.seed,
                )
                resp = (resp2 or "").strip()
                ttft_s += ttft2
                e2e_s += e2e2

            response = resp.strip()
            pred = extract_answer(response)

            # Reprompt if missing
            if pred is None:
                reprompt = "Answer with ONLY one letter: A, B, C, or D.\nFinal answer:"
                resp2, ttft2, e2e2 = stream_chat_completion(
                    client=client,
                    served_model_name=served_model_name,
                    prompt=reprompt,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=5,
                    seed=args.seed,
                )
                reprompt_used += 1
                ttft_s += ttft2
                e2e_s += e2e2
                pred = extract_answer(resp2 or "")
                response = response + "\n\n[REPROMPT]\n" + (resp2 or "")

            judge = (pred == item["answer"])

            # token counting for throughput
            try:
                n_out = len(_encode(tok, response))
            except Exception:
                n_out = 0

            lat_e2e.append(e2e_s)
            lat_ttft.append(ttft_s)
            out_tokens.append(n_out)

            # accuracy breakdown
            counts["total"] += 1
            correct["total"] += int(judge)

            d = item["difficulty"]
            if d not in ("easy", "hard"):
                d = "hard"
            counts[d] += 1
            correct[d] += int(judge)

            L = item["length"]
            if L not in ("short", "medium", "long"):
                L = "long"
            counts[L] += 1
            correct[L] += int(judge)

            # write line
            item_out = dict(item)
            item_out["response"] = response
            item_out["pred"] = pred
            item_out["judge"] = judge
            item_out["context"] = context[:1000]
            fout.write(json.dumps(item_out, ensure_ascii=False) + "\n")
            fout.flush()

            if run is not None and (counts["total"] % args.log_every == 0):
                wandb.log(
                    {
                        "progress/seen": counts["total"],
                        "eff/latency_e2e_s_mean_sofar": float(statistics.mean(lat_e2e)) if lat_e2e else float("nan"),
                        "eff/ttft_s_mean_sofar": float(statistics.mean(lat_ttft)) if lat_ttft else float("nan"),
                        "debug/reprompt_used_sofar": reprompt_used,
                    }
                )

        fout.close()

    # finalize metrics
    def acc(k: str) -> float:
        return (correct[k] / counts[k]) if counts[k] > 0 else float("nan")

    overall = acc("total")
    easy = acc("easy")
    hard = acc("hard")
    short = acc("short")
    medium = acc("medium")
    long = acc("long")

    e2e_mean = float(statistics.mean(lat_e2e)) if lat_e2e else float("nan")
    e2e_p50 = float(percentile(lat_e2e, 0.50))
    e2e_p95 = float(percentile(lat_e2e, 0.95))

    ttft_mean = float(statistics.mean(lat_ttft)) if lat_ttft else float("nan")
    ttft_p50 = float(percentile(lat_ttft, 0.50))
    ttft_p95 = float(percentile(lat_ttft, 0.95))

    total_out = float(sum(out_tokens))
    total_time = float(sum(lat_e2e))
    toks_per_s = (total_out / total_time) if total_time > 0 else float("nan")

    peak_vram = float(getattr(vs, "peak_mib_per_gpu_max", float("nan")))

    summary = {
        "acc_overall": overall,
        "acc_easy": easy,
        "acc_hard": hard,
        "acc_short": short,
        "acc_medium": medium,
        "acc_long": long,
        "e2e_mean_s": e2e_mean,
        "e2e_p50_s": e2e_p50,
        "e2e_p95_s": e2e_p95,
        "ttft_mean_s": ttft_mean,
        "ttft_p50_s": ttft_p50,
        "ttft_p95_s": ttft_p95,
        "tokens_per_s": toks_per_s,
        "peak_vram_mib": peak_vram,
        "reprompt_used": reprompt_used,
        "out_file": out_file,
    }

    print(json.dumps(summary, indent=2))

    if run is not None:
        wandb.log(
            {
                "acc/overall": overall,
                "acc/easy": easy,
                "acc/hard": hard,
                "acc/short": short,
                "acc/medium": medium,
                "acc/long": long,
                "eff/latency_e2e_s_mean": e2e_mean,
                "eff/latency_e2e_s_p50": e2e_p50,
                "eff/latency_e2e_s_p95": e2e_p95,
                "eff/ttft_s_mean": ttft_mean,
                "eff/ttft_s_p50": ttft_p50,
                "eff/ttft_s_p95": ttft_p95,
                "eff/output_tokens_total": total_out,
                "eff/tokens_per_s": toks_per_s,
                "eff/peak_vram_mib_per_gpu_max": peak_vram,
                "debug/reprompt_used_total": reprompt_used,
            }
        )
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--cache_dir", type=str,
                        default="/insomnia001/depts/edu/COMS-E6998-015/dwz2107/hf_datasets")
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-9B-Instruct")
    parser.add_argument("--cot", "-cot", action="store_true")
    parser.add_argument("--no_context", "-nc", action="store_true")
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=1)  # ignored; single-stream only
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--rep", type=int, default=int(os.environ.get("RUN_REP", "1")))
    parser.add_argument("--tokenizer_id", type=str, default=os.environ.get("TOKENIZER_ID", ""))
    parser.add_argument("--max_len", type=int, default=int(os.environ.get("CTX_LEN", "8192")))
    args = parser.parse_args()

    if args.n_proc != 1:
        print("[warn] n_proc ignored; forcing single-stream.")
        args.n_proc = 1

    run_eval(args)


if __name__ == "__main__":
    main()
