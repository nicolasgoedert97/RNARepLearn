# RNARepLearn
This package is used to train RNA sequence + structure graph embeddings. Aggregating the sequence and structural features of RNAs, we create embeddings for each nucleotide in a RNA sequence. These embeddings serve as alternative to traditional 1-hot encoding and can be used as input to RNA-related prediction tasks.

## Setup
### Setup conda environment
A conda environment file is provided `conda/env.yml`. This environment should contain all necessary package dependencies.

### RNARepLearn installation:
After activating the conda environment run:
``` 
git clone https://github.com/nicolasgoedert97/RNARepLearn.git
cd RNARepLearn
pip install .
``` 

### RFAM preprocessing
The rfam submodule can be used to download and process data from the RFAM database
``` 
cd rfam
snakemake --config-file config.yml
``` 
### Efficient loading/saving of training data
This package uses the bioio package (https://github.com/mhorlacher/bioio) to save the training data to tfrecords. 
``` 
git clone https://github.com/mhorlacher/bioio
cd bioio
pip install .
``` 
## Data preprocessing
Alternatively to the rfam module, to process training data for immediate use, the following file structure is desireable:
``` 
<Data-directory>
  |_____<Dataset1>
        |_________<Dataset1>.fa / <Dataset1>.csv
  |_____<Dataset2>
  .
  .
```
Ideally sequece information is provided in fasta or tsv format. Two snakemake workflows are provided to process the data: ´preprocess/5utr´ and ´preprocess/fold´. The ´fold´ workflow takes up fasta files and the ´5utr´ workflow takes .csv files as input, containing columns: ´seq´ for the sequence information and ´5utr´ for the mean ribosome load. These workflows both compute the secondary structure, based on the RNA sequences. For the latter, additionally the 5'UTR mean ribosome load information of each sequence is labeled for the given RNA sequence. This creates a subfolder called ´tfrecord´ for each dataset. 


``` 
<Data-directory>
  |_____<Dataset1>
        |_________<Dataset1>.fa / <Dataset1>.csv
        |_________tfrecord
                  |____<Dataset1>.tfrecord
                  |____<Dataset1>.tfrecord.features.json
                  |____<Dataset1>.index
  |_____<Dataset2>
  .
  .
```

The data can then be loaded as follows:

``` 
from RNARepLearn.datasets import GFileDataset, GFileDatasetUTR
datasets = ["<Dataset1>","<Dataset2>"]
data = GFileDataset("<Data-directory>",datasets)
```

## Example training setup
``` 
import torch
import torch_geometric

# Path to rfam base folder
rfam_dir = "../rfam/data/raw/processed/release-14.8"
rfams = ["RF00001","RF00005"]

from RNARepLearn.datasets import GFileDataset

dataset = GFileDataset(rfam_dir, rfams)

from RNARepLearn.utils import train_val_test_loaders

train_loader, val_loader, test_loader = train_val_test_loaders(dataset, 0.8, 0.1, 0.1)

from RNARepLearn.models import TestModel
model = TestModel()

from RNARepLearn.train import MaskedTraining
training = MaskedTraining(model, 10, 15, writer)

training.run(train_loader, val_loader)
``` 
## Command line excecution
Define datasets and other parameters inside config.gin
``` 
rnareplearn --gin config.gin
``` 
