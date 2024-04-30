#!/bin/sh

#SBATCH --job-name=Train-Models
#SBATCH --nodes=1 --gres=gpu:RTX_6000:4 --ntasks-per-node=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/MLP_Training-%a.out
#SBATCH --mail-type=NONE
#SBATCH --array=1-12

module load anaconda3
conda activate ml
srun python3 train_mlp_models.py --job_array_num "${SLURM_ARRAY_TASK_ID}"
