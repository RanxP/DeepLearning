export WANDB_API_KEY=99ea28c657136b69d0f2fb505191b0c5b80f05fa
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

wandb sync wandb/5993429/wandb/offline-run-20240421_111228-v1uu9yw7
wandb sync wandb/5993427/wandb/offline-run-20240421_111122-ytf3yth2
wandb sync wandb/5993425/wandb/offline-run-20240421_110956-a2ht8h2w
