#!/bin/bash
#SBATCH -JSlurmPythonExample                                 
#SBATCH --nodes=1 --ntasks-per-node=12 --gres=gpu:A100:1
#SBATCH --mem-per-gpu=32G                            
#SBATCH -t10:00:00                                     
#SBATCH -oReport-%j.out                                 
#SBATCH --mail-type=BEGIN,END,FAIL                      
#SBATCH --mail-user=rmadhusudhana3@gatech.edu                

cd $SLURM_SUBMIT_DIR                           

module load anaconda3                       
conda activate ml
python embed.py    