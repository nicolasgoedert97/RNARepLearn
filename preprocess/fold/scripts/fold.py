import pandas as pd
from Bio import SeqIO
import os
import torch
import numpy as np
import RNA
import argparse



def computeBPPM(seq):
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = np.asarray(fc.bpp())
    return basepair_probs[1:,1:]

def sequence2int_np(sequence):
    base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    return np.array([base2int.get(base, 999) for base in sequence], np.int8)

def generate_edges(bpp):
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

parser = argparse.ArgumentParser()
parser.add_argument('-base', required=True)
parser.add_argument('-rfam', required=True)
parser.add_argument('-tresh', required=True)

args = parser.parse_args()
print(args)
base, rfam, seq_leng_tresh = args.base, args.rfam, int(args.tresh)

csv = pd.read_csv(os.path.join(base,rfam,rfam+".sizes.tsv"), sep="\t",header=None)
csv = csv.sort_values(0)
csv = csv[csv[2]<seq_leng_tresh]


id_set = set(csv[1])

fasta = SeqIO.parse(os.path.join(base,rfam,rfam+".fa"), 'fasta')

illegal = 0
with open(os.path.join(base,rfam,"files.list"),"w") as files_list:
    for s in fasta:
        if s.id in id_set:
            ID = "_".join(s.id.split("/"))
            output = os.path.join(base, rfam,"pt", ID+".pt")
            if not os.path.exists(output):
                os.makedirs(os.path.dirname(output),exist_ok=True)
                
                classes = sequence2int_np(s.seq.upper())
                one_hot = one_hot_encode(classes)
                if one_hot is None:
                    illegal +=1
                    continue
                seq = torch.tensor(one_hot_encode(sequence2int_np(s.seq.upper())))
                bpp = computeBPPM(str(s.seq.upper()))
                edge_data = generate_edges(bpp)
                edges = torch.tensor(edge_data[["A","B"]].to_numpy())
                edge_weight = torch.tensor(edge_data["h_bond"].to_numpy())

                sample_dict = {"seq":seq, "edges":edges, "edge_weight":edge_weight, "rfam":rfam, "classes":classes, "id":ID}
                torch.save(sample_dict, output)
                files_list.write(ID+"\n")

