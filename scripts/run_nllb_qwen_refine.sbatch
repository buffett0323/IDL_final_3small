#!/bin/bash
#SBATCH --job-name=nllb_qwen_refine
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --chdir=/data/user_data/haolingp/IDL_final_3small
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/nllb_qwen_refine.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/nllb_qwen_refine.err

echo "Starting job"
cd /data/user_data/haolingp/IDL_final_3small
echo "CD done, pwd: $(pwd)"
source ~/.bashrc
echo "bashrc sourced"
conda activate vllm
echo "conda activated, env: $CONDA_DEFAULT_ENV"
export HF_HOME="/data/user_data/haolingp/hf_cache"
echo "HF_HOME set"
python scripts/run_nllb_qwen_refine.py
