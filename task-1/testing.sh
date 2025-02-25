#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_gpu_job
#SBATCH --output=output.log
#SBATCH --error=error.log

python task.py
