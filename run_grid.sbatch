#!/bin/sh
#
#SBATCH --account=edu                 # The account name for the job.
#SBATCH --job-name=run_grid           # The job name.
#SBATCH --gres=gpu:1                  # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 2                          # CPU cores to use.
#SBATCH --time=10:30:00                # Time
#SBATCH --mem-per-cpu=8gb             # Memory per CPU core.

set -e

mkdir -p logs
module load cuda

source /insomnia001/depts/edu/COMS-E6998-015/dwz2107/envs/ssm-venv/bin/activate

# vLLM GPU memory utilization is controlled by your vLLM flags/config (e.g. --gpu-memory-utilization 0.9)
python run_grid.py

