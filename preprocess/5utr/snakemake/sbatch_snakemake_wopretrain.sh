sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J UTR
#SBATCH --account=hai_rnareplearn
#SBATCH -p booster
#SBATCH -t 03:00:00


source $HOME/.bashrc

conda activate RL

all_args=("$@")
ds_names=("${all_args[@]:6}")



rnareplearn --gin $1 --dataset_path $2 --dataset_type UTR --train_indices $3 --val_indices $4  --output $5 --dataset_names $6 --train_mode TE --eval_model


EOF