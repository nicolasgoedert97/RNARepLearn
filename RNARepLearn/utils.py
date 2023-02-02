import random
import torch
import numpy as np
from scipy import sparse
from torch_geometric.loader import DataLoader
import pandas as pd

def mask_batch(batch, percentage):
    mask = [random.randrange(100) < percentage for i in range(batch.x.shape[0])]

    # mask nodes
    batch.x[mask] = torch.tensor([0.0,0.0,0.0,0.0],dtype=torch.float64)
    
    # mask edges
    masked_bases = np.asarray(range(len(mask)))[mask]
    masked_edges_index_from = np.where(np.isin(batch.edge_index[0],masked_bases))
    masked_edges_index_to = np.where(np.isin(batch.edge_index[1],masked_bases))
    masked_edges = np.union1d(masked_edges_index_from, masked_edges_index_to)
    
    batch.edge_weight[masked_edges]=0.0
    
    return mask, masked_edges

def edge_index_to_weight_list(batch, pred_bpp, mask):
    weightlist = []
    for x in range(len(batch.edge_index[0])):
        from_edge = batch.edge_index[0][x]
        to_edge = batch.edge_index[1][x]
        weightlist.append(pred_bpp[from_edge][to_edge])
    return weightlist

def reconstruct_bpp(edge_index, weights, shape):
    mat = sparse.coo_matrix((weights, (edge_index[0], edge_index[1])), shape=shape).toarray()
    return mat

def train_val_test_loaders(dataset, train_split, test_split, val_split, batch_size=32):
    assert sum([train_split,val_split,test_split]) == 1
    train_size = int(train_split*len(dataset))
    test_size = int(test_split * len(dataset))
    val_size = len(dataset)-train_size-test_size

    print("Training:\t"+str(train_size))
    print("Test:\t"+str(test_size))
    print("Validation:\t"+str(val_size))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(val_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)

def computeBPPM(seq):
    # fc = RNA.fold_compound(seq)
    # (propensity, ensemble_energy) = fc.pf()
    # basepair_probs = np.asarray(fc.bpp())
    # return basepair_probs[1:,1:]
    #TODO safe remove function
    return None

def sequence2int_np(sequence):
    base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    return np.array([base2int.get(base, 999) for base in sequence], np.int8)

def generate_edges(seq_len,bpp):
    X = np.zeros((seq_len,seq_len))
    X[np.triu_indices(X.shape[0], k = 1)] = bpp
    X = X+X.T

    df = pd.DataFrame(X)
    unpaired_probs = 1-df.sum(axis=1)
    np.fill_diagonal(df.values, unpaired_probs)
    adf = df.stack().reset_index()
    adf = adf.rename(columns={"level_0":"A","level_1":"B",0:"h_bond"})
    adf = adf[adf["h_bond"]!=0.0]

    # Add Covalent bonds
    # adf["cov_bond"]=0.0

    # cov_bonds=[]
    # for tupl in [(float(i),float(i+1)) for i in range(seq_len-1)]:
    #     cov_bonds.append(tupl)
    #     cov_bonds.append((tupl[1],tupl[0]))
    # conv_bonds = pd.DataFrame(cov_bonds, columns=["A","B"])
    # conv_bonds["h_bond"]=0.0
    # conv_bonds["cov_bond"]=1.0
    # adf.append(cov_bonds)

    # adf.drop_duplicates(subset = ['A', 'B'],keep = 'last').reset_index()

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