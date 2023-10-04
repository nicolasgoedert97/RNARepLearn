# %%
import argparse

import numpy as np
from tqdm import tqdm
from Bio import SeqIO

from rfampy.io import load_npy, load_npy_count
from rfampy.fold import RNAFold

# %%
folder = RNAFold()

# %%
def fold(record_dict):
    return {'RNAfold': folder(record_dict['seq'])}

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy', metavar='<rfam.npy>')
    parser.add_argument('-o', '--output', metavar='<rfam.fold.npy>')
    args = parser.parse_args()
    
    n_total = load_npy_count(args.npy)
    
    with open(args.output, 'wb') as f:
        for record_dict in tqdm(load_npy(args.npy), total=n_total):
            record_dict.update(fold(record_dict))
            np.save(file=f, arr=record_dict, allow_pickle=True)

# %%
if __name__ == '__main__':
    main()