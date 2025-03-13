#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_task1_dot_job
#SBATCH --output=output_task1_dot.log
#SBATCH --error=error_task1_dot.log

python task.py
