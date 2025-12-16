# colab_env_ssm.py
import os, sys, subprocess, re, shutil, time
from pathlib import Path

from config.pure_ssm_config import PURE_SSM_MODELS   # NEW


# ---- Pins (HPC parity) ----
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_PKGS  = ["torch==2.8.0+cu126", "torchvision==0.23.0+cu126", "torchaudio==2.8.0+cu126"]
HF_PINNED   = ["transformers==4.57.0", "datasets==3.0.2", "tokenizers==0.22.1",
               "safetensors==0.4.5", "einops==0.8.1"]
VLLM_VER    = "0.10.2"
MAMBA_VER   = "2.2.6.post3"

# ---- Models to guarantee offline (Pure SSM track) ----
MODELS = [{"repo": "facebook/opt-125m", "rev": "main"}] + [
    {
        "repo": cfg["hf_id"],
        "rev": cfg["revision"],
    }
    for cfg in PURE_SSM_MODELS.values()
]



def setup_env():
    """
    Set up the Colab environment for the pure SSM track:
      - Mount Drive
      - Set caches (pip, HF, torch extensions)
      - Install pinned Torch / HF / vLLM / mamba-ssm
      - Prefetch HF model snapshots to Drive
      - Sync HF cache to local and enable offline mode
      - Run a small vLLM smoke test

    Returns:
      DRIVE: path to Google Drive root (e.g., /content/drive/MyDrive)
    """

    # ---- Mount Drive ----
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE = "/content/drive/MyDrive"

    # ---- Drive paths ----
    PIP_CACHE       = f"{DRIVE}/.cache/pip"
    HF_HOME_DRIVE   = f"{DRIVE}/.cache/huggingface"   # canonical HF cache on Drive
    WHEELHOUSE_BASE = f"{DRIVE}/wheelhouse"           # for built wheels (e.g., mamba-ssm)
    TORCH_EXT_BASE  = f"{DRIVE}/.cache/torch_extensions"
    for d in (PIP_CACHE, HF_HOME_DRIVE, WHEELHOUSE_BASE, TORCH_EXT_BASE):
        os.makedirs(d, exist_ok=True)

    # ---- GPU -> arch ----
    gpu_name = subprocess.getoutput(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
    ).strip()
    if   "A100" in gpu_name:                               arch = "8.0"
    elif "RTX A6000" in gpu_name or "A6000" in gpu_name:  arch = "8.6"
    elif "L4" in gpu_name:                                arch = "8.9"
    elif "T4" in gpu_name:                                arch = "7.5"
    elif "V100" in gpu_name:                              arch = "7.0"
    else:                                                 arch = "8.0"
    print(f"GPU: {gpu_name or 'unknown'} | TORCH_CUDA_ARCH_LIST={arch}")

    # ---- Persistent caches on Drive; base env ----
    os.environ["PIP_CACHE_DIR"] = PIP_CACHE
    os.environ["HF_HOME"] = HF_HOME_DRIVE      # authoritative HF cache -> will sync to local for runtime
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["MAX_JOBS"]  = "8"
    os.environ["FORCE_CUDA"]= "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---- Force V0 engine in Colab (avoid V1 handshake + memory hog) ----
    os.environ["VLLM_USE_V1"] = "0"

    # vLLM runtime flags
    os.environ["VLLM_TORCH_COMPILE"] = "0"
    os.environ["VLLM_USE_CUDA_GRAPH"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    def pip(args, env=None):
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location"] + args
        subprocess.check_call(cmd, env=env or os.environ.copy())

    # ---- Base tools + Torch ----
    pip(["-U", "pip", "wheel", "packaging", "ninja", "hatchling"])
    pip(["--index-url", TORCH_INDEX] + TORCH_PKGS)

    # ---- HF libs + vLLM (official wheels) ----
    pip(HF_PINNED)
    pip(["--extra-index-url", f"https://wheels.vllm.ai/{VLLM_VER}/", f"vllm=={VLLM_VER}"])

    # ---- ABI-tagged Torch extensions dir (prevents .so reuse across stacks) ----
    import torch
    abi = f"pt{torch.__version__}-cu{torch.version.cuda}-sm{arch}-py{sys.version_info.major}{sys.version_info.minor}-vllm{VLLM_VER}"
    TORCH_EXT_DRIVE = os.path.join(TORCH_EXT_BASE, abi)
    os.makedirs(TORCH_EXT_DRIVE, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = TORCH_EXT_DRIVE

    # ---- Mamba-SSM: reuse cached wheel; else build once into Drive wheelhouse ----
    WHEELHOUSE = os.path.join(WHEELHOUSE_BASE, "mamba-ssm", abi)
    os.makedirs(WHEELHOUSE, exist_ok=True)
    have_mamba = any(fn.endswith(".whl") for fn in os.listdir(WHEELHOUSE))
    pip(["--prefer-binary", "causal-conv1d>=1.5.0.post3"])
    if have_mamba:
        pip(["--no-index", "--find-links", WHEELHOUSE, f"mamba-ssm=={MAMBA_VER}"])
    else:
        subprocess.check_call([
            sys.executable, "-m", "pip", "wheel",
            "--no-build-isolation", "-w", WHEELHOUSE, f"mamba-ssm=={MAMBA_VER}"
        ])
        pip(["--no-index", "--find-links", WHEELHOUSE, f"mamba-ssm=={MAMBA_VER}"])

    # ---- HF snapshot helpers ----
    def _hf_repo_cache_root(repo_id: str, hf_home: str) -> Path:
        org, name = repo_id.split("/", 1)
        return Path(hf_home) / "hub" / f"models--{org.replace('/', '--')}--{name.replace('/', '--')}"

    def _looks_like_commit(s: str) -> bool:
        import re as _re
        return bool(_re.fullmatch(r"[0-9a-f]{40}", s))

    def snapshot_present(repo_id: str, revision: str, hf_home: str) -> tuple[bool, str | None]:
        root = _hf_repo_cache_root(repo_id, hf_home)
        if not root.exists():
            return False, None
        if _looks_like_commit(revision):
            snap = root / "snapshots" / revision
            return snap.is_dir() and any(snap.iterdir()), revision
        ref = root / "refs" / revision
        if ref.is_file():
            commit = ref.read_text().strip()
            snap = root / "snapshots" / commit
            return snap.is_dir() and any(snap.iterdir()), commit
        return False, None

    # ---- Prefetch missing snapshots to Drive (ONLINE if needed) ----
    from huggingface_hub import snapshot_download
    missing = [m for m in MODELS if not snapshot_present(m["repo"], m["rev"], HF_HOME_DRIVE)[0]]
    if missing:
        print("\n==> Prefetching missing snapshots to Drive...")
        for m in missing:
            p = snapshot_download(
                repo_id=m["repo"],
                revision=m["rev"],
                cache_dir=os.path.join(HF_HOME_DRIVE, "hub"),
            )
            print(f"  - {m['repo']}@{m['rev']} -> {p}")
    else:
        print("\n==> All requested snapshots already present on Drive.")

    # ---- Sync Drive HF cache -> local for FAST runtime; then go fully OFFLINE ----
    HF_HOME_LOCAL = "/root/.cache/huggingface"
    os.makedirs(HF_HOME_LOCAL, exist_ok=True)
    subprocess.call([
        "bash", "-lc",
        f"shopt -s dotglob && cp -an {HF_HOME_DRIVE}/* {HF_HOME_LOCAL}/ || true"
    ])
    subprocess.call(["rsync", "-a", "--delete", f"{HF_HOME_DRIVE}/", f"{HF_HOME_LOCAL}/"])
    os.environ["HF_HOME"] = HF_HOME_LOCAL
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ---- Verify base environment ----
    import transformers, vllm
    from mamba_ssm import Mamba
    print("\n==> Environment")
    print(f"CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__} | vLLM: {vllm.__version__} | Mamba OK: {Mamba is not None}")

    # ---- Start vLLM (V0) conservatively, then you can raise the util later ----
    from vllm import LLM, SamplingParams
    GPU_UTIL_FOR_SMOKE = 0.20  # increase to 0.90 for real runs after the smoke succeeds

    print(f"\n==> vLLM V0 smoke (gpu_memory_utilization={GPU_UTIL_FOR_SMOKE})...")
    llm = LLM(
        model=MODELS[0]["repo"],
        revision=MODELS[0]["rev"],
        tokenizer_revision=MODELS[0]["rev"],
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_UTIL_FOR_SMOKE,
    )

    out = llm.generate(["hello colab-hpc"], SamplingParams(max_tokens=8))
    print(out[0].outputs[0].text)

    print("\n✅ Ready. (Drive-cached + offline; exact HPC pins on vLLM 0.10.2, V0 engine on Colab)")
    print("\n✅ colab_env_ssm.setup_env() finished.")

    
    REPO_ROOT = "/content/drive/MyDrive/SSM_experiment/pure_ssm"
    return REPO_ROOT, DRIVE
