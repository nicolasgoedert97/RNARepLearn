sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J RNARepLearn
#SBATCH --account=hai_rnareplearn
#SBATCH -p booster
#SBATCH -t 24:00:00


source $HOME/.bashrc

conda activate RL

all_args=("$@")
ds_names=("${all_args[@]:6}")



rnareplearn --gin $1 --dataset_path $2 --dataset_type RFAM --train_indices $3 --val_indices $4  --output $5 --train_mode masked

EOF