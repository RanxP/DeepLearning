export WANDB_API_KEY=99ea28c657136b69d0f2fb505191b0c5b80f05fa
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 ensamble/train.py --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes" --cloud_exec --batch_size 10 --workers 14 --number_of_epochs 35