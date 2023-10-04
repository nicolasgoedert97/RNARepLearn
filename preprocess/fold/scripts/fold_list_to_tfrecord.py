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
parser.add_argument('-base', required=True)
parser.add_argument('-files_list', required=True)
parser.add_argument('-output', required=True)
parser.add_argument('-files_len', required=True)


args = parser.parse_args()
base = args.base

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

def generate_edges(bpp):
    bpp = bpp + bpp.T
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
    def __init__(self, base, ids, length):
        self.base = base
        self.ids = open(ids, "r")
        self.pbar = tqdm(total=int(length))
        self.idx = 0

        self.rfam_fasta = None
        self.fasta_dict = {}
        self.rfam_records = []

    def __iter__(self):
        return self


    # Python 3 compatibility
    def __next__(self):
        return self.next()


    def next(self):
        rfam, _ , id = self.ids.readline().strip().split("/")
        id = id[:-3]
        id = rreplace(id,"_","/", 1)
        
        ## on new rfam
        if rfam != self.rfam_fasta:
            self.rfam_fasta = rfam
            fasta = SeqIO.parse(os.path.join(args.base,rfam,rfam+".fa"), 'fasta')
            self.fasta_dict = to_dict_remove_dups(fasta)
            
            
        record = self.fasta_dict[id]
        assert record is not None, "Fasta entry not found with ID "+id
        
        classes = sequence2int_np(record.seq.upper())
        one_hot = one_hot_encode(classes)
        ID = "_".join(record.id.split("/"))
        seq = one_hot
        bpp = computeBPPM(str(record.seq.upper()))
        edge_data = generate_edges(bpp)
        edges = edge_data[["A","B"]].to_numpy()
        edge_weight = edge_data["h_bond"].to_numpy()
        record_dict = {"seq":seq, "edges":edges, "edge_weight":edge_weight, "rfam":rfam, "classes":classes, "id":ID}
        self.idx+=1
        self.pbar.update(1)
        return record_dict

fit = FoldIterator(args.base, args.files_list, args.files_len)

os.makedirs(os.path.join(args.output, "tfrecord"), exist_ok=True)

tf_ds = bioio.tf.utils.dataset_from_iterable(fit)
bioio.tf.dataset_to_tfrecord(tf_ds, os.path.join(args.output,"tfrecord","u300.tfrecord"))
bioio.tf.index_tfrecord(os.path.join(args.output,"tfrecord","u300.tfrecord"),os.path.join(args.output,"tfrecord","u300.index"))

# rfam_fasta = None
# fasta_dict = {}

# rfam_records = []
# rfam_count = 0
# with open(args.files_list, "r") as files_list:
#     ## enum files list
#     pbar = tqdm(files_list, total=int(args.files_len))
#     for f in pbar:
#         rfam, _ , id = f.strip().split("/")
#         id = id[:-3]
#         id = rreplace(id,"_","/", 1)
        
#         ## on new rfam
#         if rfam != rfam_fasta:
#             ## if not first, write previous rfam
#             if rfam_fasta is not None:
#                 os.makedirs(os.path.join(args.output,rfam,"tfrecord"), exist_ok=True)

#                 tf_ds = bioio.tf.utils.dataset_from_iterable(rfam_records)
#                 bioio.tf.dataset_to_tfrecord(tf_ds, os.path.join(args.output,rfam,"tfrecord",rfam+".tfrecord"))
#                 bioio.tf.index_tfrecord(os.path.join(args.output,rfam,"tfrecord",rfam+".tfrecord"),os.path.join(args.output,rfam,"tfrecord",rfam+".index"))

#             rfam_records = []
#             rfam_fasta = rfam
#             fasta = SeqIO.parse(os.path.join(args.base,rfam,rfam+".fa"), 'fasta')
#             pbar.set_description(rfam)
#             fasta_dict = to_dict_remove_dups(fasta)
            

            
#         record = fasta_dict[id]
#         assert f is not None, "Fasta entry not found with ID "+id
        
#         classes = sequence2int_np(record.seq.upper())
#         one_hot = one_hot_encode(classes)
#         ID = "_".join(record.id.split("/"))
#         seq = one_hot
#         bpp = computeBPPM(str(record.seq.upper()))
#         edge_data = generate_edges(bpp)
#         edges = edge_data[["A","B"]].to_numpy()
#         edge_weight = edge_data["h_bond"].to_numpy()
#         record_dict = {"seq":seq, "edges":edges, "edge_weight":edge_weight, "rfam":rfam, "classes":classes, "id":ID}
#         rfam_records.append(record_dict)



