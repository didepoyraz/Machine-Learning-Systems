#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_ANN_cosine_job
#SBATCH --output=output_ANN_cosine.log
#SBATCH --error=error_ANN_cosine.log

# Load Conda into the shell
source ~/miniconda3/bin/activate  # Activate base environment
conda activate mlsys  # Activate your environment

# Run Python script with cosine distance metric
python task.py --test ann --distance cosine