#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def http_ok(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            status = getattr(r, "status", 0)
            return 200 <= status < 300
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


class NvidiaSmiSampler:
    """Samples GPU memory via nvidia-smi to a CSV file (no Python deps)."""

    def __init__(self, out_csv: Path, interval_ms: int = 100):
        self.out_csv = out_csv
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self._fh = None

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_csv.open("w", encoding="utf-8")
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

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

    vllm = g.get("vllm", {})
    extra: list[str] = []
    if vllm.get("extra_args"):
        extra += shlex.split(vllm["extra_args"])
    if model_cfg.get("vllm_extra_args"):
        extra += shlex.split(model_cfg["vllm_extra_args"])

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        hf_id,
        "--trust-remote-code",
        "--dtype",
        vllm.get("dtype", "bfloat16"),
        "--max-model-len",
        str(ctx),
        "--max-num-seqs",
        str(vllm.get("max_num_seqs", 1)),
        "--seed",
        str(g.get("seed", 42)),
        "--generation-config",
        vllm.get("generation_config", "vllm"),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(vllm.get("gpu_memory_utilization", 0.90)),
        "--served-model-name",
        served,
        *extra,
    ]


def run_pred(model_cfg: Dict[str, Any], g: Dict[str, Any], ctx: int, rep: int, port: int) -> int:
    served = model_cfg.get("served", model_cfg["hf_id"])
    tokenizer_id = model_cfg.get("tokenizer_id", model_cfg["hf_id"])
    family = model_cfg.get("family", served)

    env = os.environ.copy()
    env.update(
        {
            "VLLM_URL": f"http://127.0.0.1:{port}/v1",
            "CTX_LEN": str(ctx),
            "SEED": str(g.get("seed", 42)),
            "RUN_REP": str(rep),
            "TOKENIZER_ID": tokenizer_id,
        }
    )

    wandb_cfg = g.get("wandb", {})
    if wandb_cfg.get("enable", True):
        env["WANDB_PROJECT"] = wandb_cfg.get("project", "longbench-vllm")
        if wandb_cfg.get("entity"):
            env["WANDB_ENTITY"] = wandb_cfg["entity"]
        env["WANDB_GROUP"] = family
        env["WANDB_NAME"] = f"{served}__ctx{ctx}__rep{rep}"

    results_dir = (ROOT / g.get("paths", {}).get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((ROOT / "scripts" / "pred_vllm.py").resolve()),
        "-m",
        served,
        "-s",
        str(results_dir),
        "-n",
        "1",
    ]
    if wandb_cfg.get("enable", True):
        cmd.append("--wandb")

    return subprocess.call(cmd, env=env)


def kill_process_group(proc: subprocess.Popen, timeout_s: int = 10) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        return
    try:
        proc.wait(timeout=timeout_s)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", required=True, help="Path to configs/models/*.json")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--global_cfg", default=str(ROOT / "configs" / "global.json"))
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    g = read_json(Path(args.global_cfg))
    m = read_json(Path(args.model_cfg))

    port = int(args.port if args.port is not None else g.get("port", 8000))
    repeats = int(g.get("repeats", 2))

    logs_dir = (ROOT / g.get("paths", {}).get("logs_dir", "logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    served = m.get("served", m["hf_id"])
    tag = f"{served}__ctx{args.ctx}"
    server_log = logs_dir / f"{tag}.server.log"
    vram_csv = logs_dir / f"{tag}.vram.csv"

    vram_cfg = g.get("vram_sampling", {})
    sampler = NvidiaSmiSampler(vram_csv, interval_ms=int(vram_cfg.get("interval_ms", 100)))

    # Respect model engine selection (v0 => VLLM_USE_V1=0, otherwise 1)
    use_v1 = (m.get("engine") != "v0")
    server_env = os.environ.copy()
    server_env["VLLM_USE_V1"] = "1" if use_v1 else "0"
    server_env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    vllm_cmd = build_vllm_cmd(m, g, args.ctx, port)
    server_log.write_text(" ".join(shlex.quote(x) for x in vllm_cmd) + "\n\n", encoding="utf-8")

    with server_log.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            vllm_cmd,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
            env=server_env,
        )
        try:
            wait_ready(port, int(g.get("timeouts", {}).get("server_ready_seconds", 180)))

            if vram_cfg.get("enable", True):
                sampler.start()

            for rep in range(1, repeats + 1):
                rc = run_pred(m, g, args.ctx, rep, port)
                if rc != 0:
                    raise RuntimeError(f"pred_vllm failed (rc={rc}) on rep={rep}")
        finally:
            try:
                sampler.stop()
            finally:
                kill_process_group(proc)

    print(
        json.dumps(
            {
                "model": m.get("name", served),
                "ctx": args.ctx,
                "peak_vram_mib": sampler.peak_mib(),
                "server_log": str(server_log),
                "vram_csv": str(vram_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
