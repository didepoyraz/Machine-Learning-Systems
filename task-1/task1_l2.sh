#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_task1_l2_job
#SBATCH --output=output_task1_l2.log
#SBATCH --error=error_task1_l2.log

python task.py
