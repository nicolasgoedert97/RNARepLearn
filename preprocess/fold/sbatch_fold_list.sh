#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J TEST_Models
#SBATCH --account=hai_rnareplearn
#SBATCH -p booster
#SBATCH -t 6:00:00


source $HOME/.bashrc

cd /p/project/hai_rnareplearn

conda activate fold

python RNARepLearn/rfam/data/fold.2D/scripts/fold_list_to_tfrecord.py -base /p/project/hai_rnareplearn/RNARepLearn/rfam/data/raw/processed/release-14.8 -files_list /p/project/hai_rnareplearn/lustre/files.list_redacted -output small_RFAM -files_len 695605
