### Resources
- https://longbench2.github.io/
- https://pypi.org/project/mamba-ssm/
- https://www.ai21.com/blog/introducing-jamba-reasoning-3b/

Link to wandb project board: https://wandb.ai/davidwz2003-columbia-university/projects

### Installation (in a virtual environment)

Clone the repo
```bash

git clone https://github.com/andli28/SSM_experiment/
cd SSM_experiment

```

Install dependencies
```bash
pip install -r requirements.txt
```

Run specific model at context length:

```bash
python scripts/pred_one.py --model_cfg (path_to_model_config) --ctx (max_context_length)
```

Run sweep:

Set the context lengths required in /configs/globals

```bash
python run_grid.py
```

Will run all models under /config/models at the specified context lengths.

Note that for Qwen-7b, the rope scaling factor must be adjusted for the largest context being run.


###Results

If you want to log results automatically, enable wandb in the global config. Otherwise results and more details can be viewed through /results/view.ipynb


###Configs

General config values are located in /configs/global.json
Model specific configs are located in /configs/models/model.json


###Credit

We utilize some of the original code from LongBench and AdaLeval's prediction pipelines for our project


## Pure SSM Long-Context Track (Mamba) Initial tests (Located in /test_noteooks)

This repository also includes a **pure SSM (Mamba) long-context evaluation pipeline**
under the `pure_ssm/` directory. It evaluates Mamba models at 8k / 16k / 32k
context on LongBench-v2, Ada-LEval, and PG-19, and reports both quality and
efficiency.

### Models

The SSM track uses:

- `state-spaces/mamba-2.8b-hf`  (≈2.8B parameters)
- `mistralai/Mamba-Codestral-7B-v0.1`  (≈7B parameters, Mamba-2)

Both are run with **vLLM** and `mamba-ssm` using a shared decoding configuration
(greedy decoding, temperature 0.0, `max_new_tokens = 256`).

### Main notebook

The end-to-end pipeline lives in:

pure_ssm/notebooks/ssm_8k_16k_32k_final.ipynb

Typical way to run it (on Google Colab):

1. Mount Google Drive and clone this repo into:

   ```text
   /content/drive/MyDrive/SSM_experiment
   ```

2. Open `pure_ssm/notebooks/ssm_8k_16k_32k_final.ipynb` in Colab.

3. Run all cells. The notebook will:

   * Set up the environment (vLLM, `mamba-ssm`, etc.),
   * Build or load 8k / 16k / 32k prompt sets for LongBench‑v2, Ada‑LEval, and PG‑19,
   * Run pure‑SSM baselines for Mamba‑2.8B and Mamba‑Codestral‑7B at each context length,
   * Run LongBench multiple‑choice evaluation via `pure_ssm/eval_longbench_mc.py`,
   * Aggregate metrics into CSV files.

### Outputs

* **Summary CSVs** are written under:

  ```text
  pure_ssm/results/
  ```

  For example, `pure_ssm/results/ssm_8k.csv` contains quality + efficiency metrics
  for the 8k context runs.

* **Per‑run logs** (JSONL with prompts, completions, tokens/s, VRAM, etc.) are
  written to a `pure_ssm_logs/` directory in your Google Drive root:

  ```text
  /content/drive/MyDrive/pure_ssm_logs/
  ```

These outputs can be used to compare pure SSM models against the Jamba
and other baselines in this repository.
