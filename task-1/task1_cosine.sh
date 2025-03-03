#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_task1_cosine_job
#SBATCH --output=output_task1_cosine.log
#SBATCH --error=error_task1_cosine.log

python task.py
