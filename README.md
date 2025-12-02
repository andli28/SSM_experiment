### Resources
- https://longbench2.github.io/
- https://pypi.org/project/mamba-ssm/
- https://www.ai21.com/blog/introducing-jamba-reasoning-3b/

### Installation (in a virtual environment)
```bash
pip install torch
pip install vllm
pip install mamba-ssm causal-conv1d
```

### Then, serve the model
```
vllm serve ai21labs/AI21-Jamba-Reasoning-3B \
  --trust-remote-code \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype float32
```

## 2. Set Up and Run the LongBench Evaluation
You will need a second terminal to clone the LongBench repository and run its evaluation client.

### Step 1: Clone the LongBench Repository
```Bash

git clone https://github.com/THUDM/LongBench.git
cd LongBench
pip install -r requirements.txt
```

### Step 2: Configure the Evaluation Script
The LongBench repo includes a script named pred.py which is designed to send requests to an API endpoint. You need to edit this file to point to your local vLLM server.

Open the file `pred.py` in a text editor.

Look for the section defining the API base URL and API key. It will look something like this:

```Python
# Find this section (or similar) inside pred.py
api_url = "https://api.openai.com/v1/chat/completions"
api_key = "YOUR_API_KEY"
```

Modify these variables to point to your local vLLM server. vLLM uses the `v1/chat/completions` endpoint by default and doesn't require an API key (but you must pass a non-empty string).

```Python
# Change it to this:
api_url = "http://localhost:8000/v1"
api_key = "not-needed" # Pass a dummy string
```

Inside LongBench/config, you'll need to update `model2path.json` to be
```json
{
    "GLM-4-9B-Chat": "THUDM/glm-4-9b-chat",
    ...
    "glm-4-plus": "THUDM/glm-4-9b-chat",
    "Jamba-Reasoning-3B": "ai21labs/AI21-Jamba-Reasoning-3B"
}
```

and `model2maxlen.json` to be
```json
{
    "GLM-4-9B-Chat": 120000,
    ...
    "claude-3.5-sonnet-20241022": 200000,
    "Jamba-Reasoning-3B": 262144
}
```

### Step 3: Run the Inference
Once pred.py is configured, you can run the inference. The --model argument here is just a name for the output folder.

```Bash
# This will run the evaluation and save predictions in 'pred/Jamba-3B/'
python pred.py --model Jamba-Reasoning-3B
```

This script will take a long time. It iterates through the LongBench dataset, sending each long-context prompt to your vLLM server and saving the model's response.

### Step 4: Get the Final Scores
After pred.py finishes, a final script computes the scores based on the saved predictions.

```Bash
# This reads the predictions from 'pred/Jamba-3B/' and calculates the final metrics
python result.py --model Jamba-Reasoning-3B
```

## Pure SSM Long-Context Track (Mamba)

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
