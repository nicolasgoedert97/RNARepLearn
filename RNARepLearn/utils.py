import random
import torch
import numpy as np
from scipy import sparse
from torch_geometric.loader import DataLoader
import pandas as pd

def mask_batch(batch, percentage, set_zero=True, seq_zero=True, struc_zero=True):

    diag = batch.edge_index[0]==batch.edge_index[1] ##true for self edges e.g. diagonal
    eq_1 = batch.edge_weight!=1 ##true for !=1 e.g. Only for bases with unpaired prob != 1

    maskable_bases_mask = torch.logical_and(diag,eq_1) # if applied to edge_index[0] (== edge_index[1]) shows indices of nodes with self_edges != 1 
    assert (batch.edge_index[0][maskable_bases_mask] == batch.edge_index[1][maskable_bases_mask]).all()

    #select indices fit to mask
    maskable_bases = batch.edge_index[0][maskable_bases_mask]

    #select X% to mask
    maskable_bases = maskable_bases[[random.randrange(100) < percentage for i in range(len(maskable_bases))]]


    # create bool vector of len(nodes) -> mask
    node_ids = np.asarray(range(batch.x.shape[0]))
    mask = np.isin(node_ids, maskable_bases.cpu())

    if set_zero:
        # if set_zero the masked bases are set to zero. Else the mask just defines on which positions to test/train/val
        masked_bases = node_ids[mask]
        masked_edges_index_from = np.where(np.isin(batch.edge_index[0],masked_bases))
        masked_edges_index_to = np.where(np.isin(batch.edge_index[1],masked_bases))
        masked_edges = np.union1d(masked_edges_index_from, masked_edges_index_to)
        
        if (seq_zero):
            batch.x[mask] = torch.tensor([0.0,0.0,0.0,0.0],dtype=torch.float64)
        if(struc_zero):
            batch.edge_weight[masked_edges]=0.0
    
    return mask

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

def train_val_test_loaders(dataset, train_split, val_split, test_split, batch_size=32):
    assert sum([train_split,val_split,test_split]) == 1
    train_size = int(train_split*len(dataset))
    test_size = int(test_split * len(dataset))
    val_size = len(dataset)-train_size-test_size

    print("Training:\t"+str(train_size))
    print("Test:\t"+str(test_size))
    print("Validation:\t"+str(val_size))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(val_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)


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

def save_dataset(indices, output):
    with open(output, "w") as datsets_file:
        datsets_file.write('\n'.join(str(i) for i in indices))



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("Earlystop!")
                return True
        return False