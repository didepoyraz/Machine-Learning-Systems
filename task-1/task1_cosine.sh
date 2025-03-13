#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_task1_cosine_job
#SBATCH --output=output_task1_cosine.log
#SBATCH --error=error_task1_cosine.log

# Load Conda into the shell
source ~/miniconda3/bin/activate  # Activate base environment
conda activate mlsys  # Activate your environment

# Run Python script
python task.py
