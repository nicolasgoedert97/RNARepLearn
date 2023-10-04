import argparse
from RNARepLearn.datasets import GFileDataset
from RNARepLearn.utils import parse_indices
import torch
import os
from torch.utils.data import Subset, random_split
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--base')
parser.add_argument('--rfam', nargs='+')
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--output")
parser.add_argument("--k_fold")

args = parser.parse_args()

ds = GFileDataset(args.base, args.rfam)
print("Original:\n"+str(ds))

os.makedirs(args.output, exist_ok=True)

if args.k_fold is not None:

        ds = Subset(ds,parse_indices(os.path.join(args.output,"train.indices")))
        print("Training set:\n"+str(ds))
        n_folds = int(args.k_fold)

        if n_folds==1:
                fold_path = os.path.join(args.output,"folds", "fold0")
                os.makedirs(fold_path, exist_ok=True)

                train, val = torch.utils.data.random_split(ds, [float(args.train),float(args.test)])

                with open(os.path.join(fold_path,"train.indices"), "w") as datsets_file:
                        datsets_file.write('\n'.join(str(i) for i in train.indices))

                with open(os.path.join(fold_path,"val.indices"), "w") as datsets_file:
                        datsets_file.write('\n'.join(str(i) for i in val.indices))
        else:
                split = KFold(n_folds, shuffle=True)
                splits = split.split(ds)
                for i, (train_set, val_set) in enumerate(splits):

                        fold_path = os.path.join(args.output,"folds", "fold"+str(i))
                        os.makedirs(fold_path, exist_ok=True)

                        train = Subset(ds, train_set)
                        val = Subset(ds, val_set)

                        with open(os.path.join(fold_path,"train.indices"), "w") as datsets_file:
                                datsets_file.write('\n'.join(str(i) for i in train.indices))

                        with open(os.path.join(fold_path,"val.indices"), "w") as datsets_file:
                                datsets_file.write('\n'.join(str(i) for i in val.indices))

else:
        train, test = torch.utils.data.random_split(ds, [float(args.train),float(args.test)])


        with open(os.path.join(args.output,"train.indices"), "w") as datsets_file:
                datsets_file.write('\n'.join(str(i) for i in train.indices))

        with open(os.path.join(args.output,"test.indices"), "w") as datsets_file:
                datsets_file.write('\n'.join(str(i) for i in test.indices))


