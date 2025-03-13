#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_task1_manhattan_job
#SBATCH --output=output_task1_manhattan.log
#SBATCH --error=error_task1_manhattan.log

python task.py
