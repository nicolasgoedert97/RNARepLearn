import argparse
import RNA
import pandas
import pandas as pd
from Bio import SeqIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import RNA
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import bioio
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-file', required=True)
parser.add_argument('-output', required=True)
parser.add_argument('-undirected', action="store_true")

args = parser.parse_args()

directed = not args.undirected

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def computeBPPM(seq):
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = np.asarray(fc.bpp())
    return basepair_probs[1:,1:]

def sequence2int_np(sequence):
    base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    return np.array([base2int.get(base, 999) for base in sequence], np.int8)

def generate_edges(bpp, directed=True):
    if directed:
        bpp = bpp+bpp.T
    
    df = pd.DataFrame(bpp)
    unpaired_probs = 1-df.sum(axis=1)
    np.fill_diagonal(df.values, unpaired_probs)
    adf = df.stack().reset_index()
    adf = adf.rename(columns={"level_0":"A","level_1":"B",0:"h_bond"})
    adf = adf[adf["h_bond"]!=0.0]

    return adf

def one_hot_encode(seq):
        nuc_d = {0:[1.0,0.0,0.0,0.0],
                 1:[0.0,1.0,0.0,0.0],
                 2:[0.0,0.0,1.0,0.0],
                 3:[0.0,0.0,0.0,1.0],
                 -25:[0.0,0.0,0.0,0.0]} ##for bases other than ATGCU in rfam sequences / these sequences are removed from the dataset
        vec=np.array([nuc_d[x] for x in seq])
        if [0.0,0.0,0.0,0.0] in vec.tolist():
            return None
        else:
            return vec
def to_dict_remove_dups(sequences):
    return {record.id: record for record in sequences}

class FoldIterator():
    def __init__(self, file, directed=True):
        print(directed)
        self.file = pd.read_csv(file)
        self.len = len(self.file)
        self.file = self.file.iterrows()
        self.pbar = tqdm(total=self.len)
        self.idx = 0
        self.directed = directed

    def __iter__(self):
        return self


    # Python 3 compatibility
    def __next__(self):
        return self.next()


    def next(self):
        id, row = next(self.file)
        seq = str(row["utr"])
        rl = row["rl"]
        
        classes = sequence2int_np(seq.upper())
        one_hot = one_hot_encode(classes)
        ID = id
        bpp = computeBPPM(seq.upper())
        seq = one_hot
        edge_data = generate_edges(bpp)
        edges = edge_data[["A","B"]].to_numpy()
        edge_weight = edge_data["h_bond"].to_numpy()
        record_dict = {"seq":seq, "edges":edges, "edge_weight":edge_weight,"rl":rl, "classes":classes, "id":ID}
        self.idx+=1
        self.pbar.update(1)
        # if self.idx % 1000 == 0:
        #     print(self.idx)
        return record_dict

fit = FoldIterator(args.file, directed)

os.makedirs(os.path.join(args.output, "tfrecord"), exist_ok=True)

tf_ds = bioio.tf.utils.dataset_from_iterable(fit)
ds_name = os.path.basename(args.output)[:-4]
bioio.tf.dataset_to_tfrecord(tf_ds, os.path.join(args.output,"tfrecord",ds_name+".tfrecord"))
bioio.tf.index_tfrecord(os.path.join(args.output,"tfrecord",ds_name+".tfrecord"),os.path.join(args.output,"tfrecord",ds_name+".index"))


