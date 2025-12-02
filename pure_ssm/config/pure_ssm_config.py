# config/pure_ssm_config.py

# Pure SSM models you will run
PURE_SSM_MODELS = {
    "mamba-2.8b": {
        "hf_id": "state-spaces/mamba-2.8b-hf",
        "revision": "main",   # later: pin to commit SHA
        "params_b": 2.8,
        "max_context": 32768,
    },
    "mamba-codestral-7b": {
        "hf_id": "mistralai/Mamba-Codestral-7B-v0.1",
        "revision": "main",
        "params_b": 7.0,
        "max_context": 32768,
    },
}

# Context lengths you care about
PURE_SSM_CONTEXTS = [8192, 16384, 32768]

# Dataset slots for your track (you’ll fill tasks later)
PURE_SSM_DATASETS = {
    "longbench_v2": {
        "tasks": [],   # e.g. ["narrative_qa", "multi_news", ...]
        "split": "validation",
    },
    "ada_leval": {
        "tasks": [],   # e.g. ["long_retrieval", "multi_doc_reasoning"]
        "split": "test",
    },
    "pg19": {
        "enabled": False,
        "split": "test",
    },
}

# Shared decoding / fairness config
DECODE_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "batch_size": 1,
}
