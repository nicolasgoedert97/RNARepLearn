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



parser = argparse.ArgumentParser()
parser.add_argument('-base', required=True)
parser.add_argument('-rfam', required=True)
parser.add_argument('-tresh', required=True)

args = parser.parse_args()
base, rfam, seq_leng_tresh = args.base, args.rfam, int(args.tresh)

def computeBPPM(seq):
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = np.asarray(fc.bpp())
    return basepair_probs[1:,1:]

def sequence2int_np(sequence):
    base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    return np.array([base2int.get(base, 999) for base in sequence], np.int8)

def generate_edges(bpp):
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

class FoldIterator():
    def __init__(self, base, rfam, seq_len):
        self.base = base
        self.rfam = rfam
        self.fasta = self.load_fasta()
        self.seq_len_lim = seq_len
        
    
    def load_fasta(self):
        return SeqIO.parse(os.path.join(self.base,self.rfam,self.rfam+".fa"), 'fasta')
        

    def __iter__(self):
        self.fasta = self.load_fasta()
        return self


    # Python 3 compatibility
    def __next__(self):
        return self.next()


    def next(self):
        valid = False
        
        record = None
        classes = None
        one_hot = None
        while not valid:
            record = next(self.fasta)
            classes = sequence2int_np(record.seq.upper())
            one_hot = one_hot_encode(classes)

            if len(classes)<=self.seq_len_lim and one_hot is not None:
                valid = True

        ID = "_".join(record.id.split("/"))
        seq = one_hot
        bpp = computeBPPM(str(record.seq.upper()))
        edge_data = generate_edges(bpp)
        edges = edge_data[["A","B"]].to_numpy()
        edge_weight = edge_data["h_bond"].to_numpy()

        record_dict = {"seq":seq, "edges":edges, "edge_weight":edge_weight, "rfam":rfam, "classes":classes, "id":ID}
        return record_dict

output_path = os.path.join(base,rfam,"tfrecord")
os.makedirs(output_path, exist_ok=True)

fit = FoldIterator(base, rfam, 300)

tf_ds = bioio.tf.utils.dataset_from_iterable(iter(fit))
bioio.tf.dataset_to_tfrecord(tf_ds, os.path.join(output_path,rfam+".tfrecord"))
bioio.tf.index_tfrecord(os.path.join(output_path,rfam+".tfrecord"),os.path.join(output_path,rfam+".index"))