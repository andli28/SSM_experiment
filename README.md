# HPML Project: An Analysis of Long-Context Scaling With State Space Machines

## Team Information
- **Team Name**: Team DAY
- **Members**:
  - Andrew Li (ayl2159)
  - David Zhang (dwz2107)
  - Yuqi Zhang (yz5072)
  

## 1. Problem Statement
Evaluate long-context language models and state space models (SSMs) on LongBench-v2, Ada-LEval, and PG-19, focusing on quality and efficiency across large context lengths (up to 131k+ for Transformers and 32k for SSM baselines).

## 2. Model Description
- Architectures: Transformer LLMs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Jamba-3B) and SSMs (Mamba-2.8B, Mamba-Codestral-7B).
- Frameworks: PyTorch with vLLM; `mamba-ssm` for SSM kernels.
- Customization: Context-length scaling via per-model JSON configs; shared greedy decoding (`temperature=0.1`, `max_new_tokens=256`, batch_size=16 for efficient serving); note Qwen-7B requires YARN rope-scaling adjustment for largest contexts.

## 3. Final Results Summary

### Key Findings

We evaluated four representative models across increasing context lengths (8k to 131k tokens):

**Accuracy Scaling with Context Length**

![Accuracy vs Context Length](overall_accuracy_ctx_lengths.png)

Among evaluated models, **Qwen2.5-7B-Instruct** demonstrates the strongest positive scaling behavior, with accuracy steadily increasing as context length grows—indicating effective use of additional context. **Llama-3.1-8B-Instruct** maintains relatively stable accuracy with slight degradation at the longest context lengths. Both **mamba-codestral-7B** and **nemotron-h-8b-r128k** underperform Transformers in absolute accuracy. The pure SSM model shows the lowest accuracy overall with limited improvement from extended context, suggesting architectural limitations in long-context reasoning. The hybrid Nemotron model achieves intermediate accuracy with a modest peak at mid-range contexts.

**Null Rate and Output Stability**

![Null Rate vs Context Length](null_output_ctx_lengths.png)

Null prediction rates (indicating unparseable or non-compliant outputs) generally increase with context length. Transformer models maintain relatively controlled null rates, with Qwen2.5-7B showing particularly low rates overall. In contrast, SSM-based and hybrid models exhibit consistently higher null rates across all context lengths, with rates increasing further as context grows. This indicates that a significant fraction of errors in smaller SSM/hybrid models stem from output-format and instruction-following failures rather than purely incorrect reasoning.

| Metric | Value |
|----------------------|-------------|
| Models Evaluated | Llama-3.1-8B, Qwen2.5-7B, Mamba-Codestral-7B, Nemotron-8B |
| Context Range | 8k–131k tokens |
| Best Performer (Accuracy) | Qwen2.5-7B-Instruct (34% @ 131k ctx) |
| Benchmark | LongBench-v2 |
| Device | NVIDIA A6000 (48 GB VRAM) |

Refer to JSONL logs in `results/` and `pure_ssm/outputs/pure_ssm_logs/` for detailed run-level metrics.

## 4. Reproducibility Instructions
### A. Requirements
Clone and install:
```bash
git clone https://github.com/andli28/SSM_experiment
cd SSM_experiment
pip install -r requirements.txt
```

### B. Wandb Dashboard
Training/eval metrics (optional logging): https://api.wandb.ai/links/davidwz2003-columbia-university/57zty5qi

### C. Specify for Training or For Inference or if Both
This repo focuses on inference/evaluation sweeps. For a single model/context run:
```bash
python scripts/pred_one.py --model_cfg configs/models/<model>.json --ctx <max_context_length>
```

To sweep all models at configured context lengths:
```bash
python scripts/run_grid.py
```
Set global contexts in `configs/global.json`; per-model settings in `configs/models/*.json`.

### D. Evaluation
Transformer results are stored under `results/*.jsonl`. For SSM baselines (8k/16k/32k) run the Colab-style pipeline:
```bash
# Open and run all cells
pure_ssm/notebooks/ssm_8k_16k_32k_final.ipynb
```
Logs land in Google Drive `pure_ssm_logs/` and summaries in `pure_ssm/results/`.

### E. Quickstart: Minimum Reproducible Result
To reproduce a transformer sweep (example):
```bash
# 1) Install
pip install -r requirements.txt
# 2) Configure contexts
python - <<'PY'
import json
path = 'configs/global.json'
cfg = json.load(open(path))

json.dump(cfg, open(path, 'w'), indent=2)
PY
# 3) Run sweep
python scripts/run_grid.py
# 4) Inspect outputs
ls results/
```

To reproduce a pure SSM 8k/16k/32k run (Colab):
```bash
# 1) Clone into Drive (in Colab)
git clone https://github.com/andli28/SSM_experiment /content/drive/MyDrive/SSM_experiment
# 2) Open and run
pure_ssm/notebooks/ssm_8k_16k_32k_final.ipynb
```

## 5. Notes
- Per-model configs: `configs/models/*.json`; globals: `configs/global.json`.
- **Results Analysis**: `results/view.ipynb` generates detailed analysis of locally stored transformer runs, including raw metric values, prediction statistics, and accuracy plots across context lengths and models. For SSM baselines, summaries are available in `pure_ssm/results/*.csv`.
- Resources: https://longbench2.github.io/, https://pypi.org/project/mamba-ssm/, https://www.ai21.com/blog/introducing-jamba-reasoning-3b/
  - Contact: [ayl2159@columbia.edu](mailto:ayl2159@columbia.edu)

