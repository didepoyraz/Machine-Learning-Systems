#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_KNN_l2_job
#SBATCH --output=output_KNN_l2.log
#SBATCH --error=error_KNN_l2.log

# Load Conda into the shell
source ~/miniconda3/bin/activate  # Activate base environment
conda activate mlsys  # Activate your environment

# Run Python script with cosine distance metric
python task.py --test knn --distance l2
