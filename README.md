# RNARepLearn
This package is used to train RNA sequence + structure graph embeddings. Aggregating the sequence and structural features of RNAs, we want to leverage the performance of downstream prediction tasks.

## Setup
### RNARepLearn installation:
``` 
git clone https://github.com/nicolasgoedert97/RNARepLearn.git
cd RNARepLearn
git setup .
``` 
Note, that some issues with exporting the conda environment exist, thus there is no valid env.yml file in this repo yet. An environment with python >3.7 pytorch, torch_gemetric, numpy and pandas installed should suffice to run the package.

### Rfam preprocessing
Preprocess rfam database. Seq-Length + Seq_number_per_family limit can be set in config.
``` 
cd rfam
snakemake --config-file config.yml
``` 

## Example training setup
``` 
import torch
import torch_geometric

# Path to rfam base folder
rfam_dir = "../rfam/data/raw/processed/release-14.8"
rfams = ["RF00001","RF00005"]

from RNARepLearn.datasets import CombinedRfamDataset, SingleRfamDataset

dataset = CombinedRfamDataset(rfam_dir, rfams , "RF00001_RF00005")

from RNARepLearn.utils import train_val_test_loaders

train_loader, val_loader, test_loader = train_val_test_loaders(dataset, 0.8, 0.1, 0.1)

from RNARepLearn.models import TestModel
model = TestModel()

from RNARepLearn.train import MaskedTraining
training = MaskedTraining(model, 10, 15, writer)

training.run(train_loader, val_loader)
``` 
