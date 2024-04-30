#!/bin/sh

#SBATCH --job-name=Extract-Embeddings
#SBATCH --nodes=1 --gres=gpu:A100:1
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=logs/Embeddings.out
#SBATCH --mail-type=NONE

module load anaconda3
conda activate mlcb_project

python3 extract_embeddings.py
