#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=0:30:00

cd /gpfs/home6/scur0773/DeepLearning/
mkdir wandb/$SLURM_JOBID

srun apptainer exec --nv /gpfs/work5/0/jhstue005/JHS_data/5lsm0_v1.sif python3 visualize_distributions_one.py