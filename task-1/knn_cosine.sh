#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_KNN_cosine_job
#SBATCH --output=output_KNN_cosine.log
#SBATCH --error=error_KNN_cosine.log

# Load Conda into the shell
source ~/miniconda3/bin/activate  # Activate base environment
conda activate mlsys  # Activate your environment

# Run Python script with cosine distance metric
python task.py --test knn --distance cosine
