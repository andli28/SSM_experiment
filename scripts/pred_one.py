#!/usr/bin/env python3
import argparse, json, os, shlex, signal, subprocess, sys, time
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def http_ok(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def wait_ready(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if http_ok(url):
            return
        time.sleep(1)
    raise RuntimeError(f"vLLM not ready after {timeout_s}s: {url}")

class VramSampler:
    """Samples GPU memory via nvidia-smi to a CSV file. No Python deps."""
    def __init__(self, out_csv: Path, interval_ms: int = 100):
        self.out_csv = out_csv
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        f = self.out_csv.open("w", encoding="utf-8")
        # keep handle for process lifetime
        self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
        self._fh = f

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self._fh.close()
        except Exception:
            pass
        self.proc = None

    def peak_mib(self) -> int:
        if not self.out_csv.exists():
            return 0
        peak = 0
        for line in self.out_csv.read_text(encoding="utf-8").splitlines():
            # format: timestamp, index, memory.used
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    peak = max(peak, int(parts[2]))
                except Exception:
                    pass
        return peak

def build_vllm_cmd(model_cfg: Dict[str, Any], g: Dict[str, Any], ctx: int, port: int) -> list:
    hf_id = model_cfg["hf_id"]
    served = model_cfg.get("served", hf_id)

    vllm = g["vllm"]
    dtype = vllm.get("dtype", "bfloat16")
    gpu_mem = str(vllm.get("gpu_memory_utilization", 0.90))
    max_num_seqs = str(vllm.get("max_num_seqs", 1))
    gen_cfg = vllm.get("generation_config", "vllm")

    extra = []
    # global extra args
    if vllm.get("extra_args"):
        extra += shlex.split(vllm["extra_args"])
    # per-model extra args
    if model_cfg.get("vllm_extra_args"):
        extra += shlex.split(model_cfg["vllm_extra_args"])

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", hf_id,
        "--trust-remote-code",
        "--dtype", dtype,
        "--max-model-len", str(ctx),
        "--max-num-seqs", max_num_seqs,
        "--seed", str(g["seed"]),
        "--generation-config", gen_cfg,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", gpu_mem,
        "--served-model-name", served,
        *extra,
    ]
    return cmd

def run_pred(model_cfg: Dict[str, Any], g: Dict[str, Any], ctx: int, rep: int, port: int) -> int:
    served = model_cfg.get("served", model_cfg["hf_id"])
    tokenizer_id = model_cfg.get("tokenizer_id", model_cfg["hf_id"])
    family = model_cfg.get("family", served)

    env = os.environ.copy()
    env["VLLM_URL"] = f"http://127.0.0.1:{port}/v1"
    env["CTX_LEN"] = str(ctx)
    env["SEED"] = str(g["seed"])
    env["RUN_REP"] = str(rep)
    env["TOKENIZER_ID"] = tokenizer_id

    if g["wandb"].get("enable", True):
        env["WANDB_PROJECT"] = g["wandb"].get("project", "longbench-vllm")
        if g["wandb"].get("entity"):
            env["WANDB_ENTITY"] = g["wandb"]["entity"]
        env["WANDB_GROUP"] = family
        env["WANDB_NAME"] = f"{served}__ctx{ctx}__rep{rep}"

    results_dir = (ROOT / g["paths"].get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str((ROOT / "scripts" / "pred_vllm.py").resolve()),
        "-m", served,
        "-s", str(results_dir),
        "-n", "1",
    ]
    if g["wandb"].get("enable", True):
        cmd.append("--wandb")

    return subprocess.call(cmd, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", required=True, help="Path to configs/models/*.json")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--global_cfg", default=str(ROOT / "configs" / "global.json"))
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    g = load_json(Path(args.global_cfg))
    m = load_json(Path(args.model_cfg))

    port = args.port if args.port is not None else int(g.get("port", 8000))
    repeats = int(g.get("repeats", 2))
    logs_dir = (ROOT / g["paths"].get("logs_dir", "logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{m.get('served', m['hf_id'])}__ctx{args.ctx}"
    server_log = logs_dir / f"{tag}.server.log"
    vram_csv = logs_dir / f"{tag}.vram.csv"

    vram_cfg = g.get("vram_sampling", {})
    sampler = VramSampler(vram_csv, interval_ms=int(vram_cfg.get("interval_ms", 100)))


    # force v0 engine for SSM-only models
    if m.get("engine") == "v0":
        os.environ["VLLM_USE_V1"] = "0"
    else: 
        os.environ["VLLM_USE_V1"] = "1"

    print(os.environ["VLLM_USE_V1"])
    server_env = os.environ.copy()


    vllm_cmd = build_vllm_cmd(m, g, args.ctx, port)
    # log the exact command for reproducibility
    server_log.write_text(" ".join(shlex.quote(x) for x in vllm_cmd) + "\n\n", encoding="utf-8")

    with server_log.open("a", encoding="utf-8") as lf:
        server_env["VLLM_USE_V1"] = "0"
        proc = subprocess.Popen(vllm_cmd, stdout=lf, stderr=lf, start_new_session=True, env = server_env)
        
        def cleanup():
            try:
                sampler.stop()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass

        try:
            wait_ready(port, int(g["timeouts"].get("server_ready_seconds", 180)))
            if vram_cfg.get("enable", True):
                sampler.start()

            for rep in range(1, repeats + 1):
                rc = run_pred(m, g, args.ctx, rep, port)
                if rc != 0:
                    raise RuntimeError(f"pred_vllm failed (rc={rc}) on rep={rep}")

        finally:
            cleanup()

    peak = sampler.peak_mib()
    print(json.dumps({
        "model": m.get("name", m.get("served", m["hf_id"])),
        "ctx": args.ctx,
        "peak_vram_mib": peak,
        "server_log": str(server_log),
        "vram_csv": str(vram_csv),
    }, indent=2))

if __name__ == "__main__":
    main()
