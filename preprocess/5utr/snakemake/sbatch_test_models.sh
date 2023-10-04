#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J TEST_Models
#SBATCH --account=hai_rnareplearn
#SBATCH -p booster
#SBATCH -t 6:00:00


source $HOME/.bashrc

conda activate RL


python scripts/test_models.py
