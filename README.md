### Resources
- https://longbench2.github.io/
- https://pypi.org/project/mamba-ssm/

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
api_url = "http://localhost:8000/v1/chat/completions"
api_key = "not-needed" # Pass a dummy string
```

You will also need to set the model name inside the script to match what vLLM is serving. In pred.py, find the function that builds the request payload (likely get_answer or similar) and ensure the "model" field matches your model ID.

```Python
# Inside the request payload/json
payload = {
    "model": "ai21labs/AI21-Jamba-Reasoning-3B", # Make sure this matches!
    "messages": ...,
    ...
}
```

### Step 3: Run the Inference
Once pred.py is configured, you can run the inference. The --model argument here is just a name for the output folder.

```Bash
# This will run the evaluation and save predictions in 'pred/Jamba-3B/'
python pred.py --model Jamba-3B 
```

This script will take a long time. It iterates through the LongBench dataset, sending each long-context prompt to your vLLM server and saving the model's response.

### Step 4: Get the Final Scores
After pred.py finishes, a final script computes the scores based on the saved predictions.

```Bash
# This reads the predictions from 'pred/Jamba-3B/' and calculates the final metrics
python result.py --model Jamba-3B
```