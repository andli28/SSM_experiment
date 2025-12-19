#!/usr/bin/env python3
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global_cfg", default=str(ROOT / "configs" / "global.json"))
    ap.add_argument("--models_dir", default=str(ROOT / "configs" / "models"))
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional model names to run (e.g., llama-8b qwen-7b)",
    )
    args = ap.parse_args()

    g = load_json(Path(args.global_cfg))
    contexts: List[int] = list(map(int, g.get("contexts", [8192, 16384, 32768])))

    models_dir = Path(args.models_dir)
    model_cfgs = sorted(models_dir.glob("*.json"))

    if args.only:
        want = set(args.only)
        model_cfgs = [p for p in model_cfgs if load_json(p).get("name") in want]

    if not model_cfgs:
        raise SystemExit("No model configs found.")

    for ctx in contexts:
        for cfg in model_cfgs:
            cmd = [
                sys.executable,
                str((ROOT / "scripts" / "pred_one.py").resolve()),
                "--model_cfg",
                str(cfg),
                "--ctx",
                str(ctx),
                "--global_cfg",
                str(Path(args.global_cfg).resolve()),
            ]
            print("\n=== RUN ===")
            print(" ".join(cmd))
            rc = subprocess.call(cmd)
            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()
