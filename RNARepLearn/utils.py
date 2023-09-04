import random
import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold
import numpy as np
from scipy import sparse
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.utils import to_dense_batch
import pandas as pd
import warnings

def parse_indices(index_path):
    with open(index_path, "r") as idxs:
        indices = idxs.readlines()
        indices = [int(i.strip()) for i in indices]
        return indices

def add_backbone(batch, add_attributes=True):
    offset = 0
    output = []
    for batch_i in range(len(batch.ID)):
        edges = []
        seq = batch[batch_i]
        for i in range(seq.x.shape[0]-1):
            i+=offset
            e1 = (i, i+1)
            e2 = (i+1, i)
            edges.append(e1)
            edges.append(e2)
        output.append(torch.tensor(edges))
        offset=i+2
    
    bb_edges = torch.cat(output).T

    if add_attributes:
        bp_attributes = torch.unsqueeze(batch.edge_weight, dim=1)
        bp_attributes = torch.cat([bp_attributes, torch.zeros_like(bp_attributes)], dim=1)
        bp_attributes = torch.tensor([[0,1]]*len(batch.edge_index[0]))

        bb_attributes = torch.tensor([[0,1]]*bb_edges.shape[1])
        bb_weights = torch.tensor([1]*bb_edges.shape[1])

        attr = torch.cat([bp_attributes, bb_attributes])
    

    batch.edge_index = torch.cat([batch.edge_index, bb_edges], dim=1)
    batch.edge_weight = torch.cat([batch.edge_weight, bb_weights])
    batch.edge_attr = attr.T

    return batch

def add_backbone_single(x, edge_index, edge_weight, add_attributes=True):
    
    edges = []
    for i in range(x.shape[0]-1):
        e1 = (i, i+1)
        e2 = (i+1, i)
        edges.append(e1)
        edges.append(e2)
    bb_edges = torch.tensor(edges).T

    if add_attributes:
        bp_attributes = torch.unsqueeze(edge_weight, dim=1)
        bp_attributes = torch.cat([bp_attributes, torch.zeros_like(bp_attributes)], dim=1)
        
        bb_attributes = torch.tensor([[0,1]]*bb_edges.shape[1])
        bb_weights = torch.tensor([1]*bb_edges.shape[1])

        attr = torch.cat([bp_attributes, bb_attributes])
    

    edge_index = torch.cat([edge_index, bb_edges], dim=1)
    edge_weight = torch.cat([edge_weight, bb_weights])
    edge_attr = attr

    return edge_index, edge_weight, edge_attr

def mask_batch(batch, percentage, set_zero=True, seq_zero=True, struc_zero=True):

    num_masked_bases = int((float(percentage)/100)*len(batch.x))
    node_ids = np.asarray(range(batch.x.shape[0]))

    diag = batch.edge_index[0]==batch.edge_index[1] ##true for self edges e.g. diagonal
    eq_1 = batch.edge_weight!=1 ##true for !=1 e.g. Only for bases with unpaired prob != 1

    maskable_bases_mask = torch.logical_and(diag,eq_1) # if applied to edge_index[0] (== edge_index[1]) shows indices of nodes with self_edges != 1 
    assert (batch.edge_index[0][maskable_bases_mask] == batch.edge_index[1][maskable_bases_mask]).all()
        
    #select indices fit to mask
    maskable_bases = batch.edge_index[0][maskable_bases_mask]

    if (len(maskable_bases) < num_masked_bases):
        still_to_mask = num_masked_bases - len(maskable_bases)
        yet_unmasked = torch.tensor(node_ids[~np.isin(node_ids,maskable_bases)])
        maskable_bases = torch.cat([maskable_bases, yet_unmasked[torch.randperm(len(yet_unmasked))][:still_to_mask]], dim=0)
    else:
        #select X% to mask
        maskable_bases = maskable_bases[torch.randperm(len(maskable_bases))[:num_masked_bases]]


    # create bool vector of len(nodes) -> mask
    mask = np.isin(node_ids, maskable_bases.cpu())

    if set_zero:
        # if set_zero the masked bases are set to zero. Else the mask just defines on which positions to test/train/val
        masked_bases = node_ids[mask]
        masked_edges_index_from = np.where(np.isin(batch.edge_index[0],masked_bases))
        masked_edges_index_to = np.where(np.isin(batch.edge_index[1],masked_bases))
        masked_edges = np.union1d(masked_edges_index_from, masked_edges_index_to)
        
        if (seq_zero):
            batch.x[mask] = torch.tensor([0.0]*batch.x.shape[1],dtype=torch.float64)
        if(struc_zero):
            batch.edge_weight[masked_edges]=0.0
            if batch.edge_attr is not None:
                batch.edge_attr[masked_edges]=torch.tensor([0.0,0.0], dtype=torch.double)
    
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


def random_train_val_test_loaders(dataset, train_split, val_split, test_split, batch_size=32, num_workers=4, pin_memory=True):
    assert sum([train_split,val_split,test_split]) == 1
    train_size = int(train_split*len(dataset))
    test_size = int(test_split * len(dataset))
    val_size = len(dataset)-train_size-test_size

    print("Training:\t"+str(train_size))
    print("Test:\t"+str(test_size))
    print("Validation:\t"+str(val_size))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True), DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True), DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory )

def indexed_train_val_test_loaders(dataset, train_indices, val_indices, test_indices, batch_size=32, num_workers=4, parallel=False):
    output = []
    if train_indices is not None:
        train_indices = parse_indices(train_indices)
        train_set = Subset(dataset, train_indices)
        if parallel:
            output.append(DataListLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True))
        else:
            output.append(DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True))
        print("Training:\t"+str(len(train_set)))
    else:
        output.append(None)

    if val_indices is not None:
        val_indices = parse_indices(val_indices)
        val_set = Subset(dataset, val_indices)
        if parallel:
            val_loader = DataListLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        else:
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        print("Validation:\t"+str(len(val_set)))
    else:
        val_loader = None
    output.append(val_loader)
    
    if test_indices is not None:
        test_indices = parse_indices(test_indices)
        test_set = Subset(dataset, test_indices)
        if parallel:
            test_loader = DataListLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        else:
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        print("Test:\t"+str(len(test_set)))
    else:
        test_loader = None
    output.append(test_loader)
    
    return output


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

    return adf

def generate_edges(bpp):
    df = pd.DataFrame(bpp)
    adf = df.stack().reset_index()
    adf = adf.rename(columns={"level_0":"A","level_1":"B",0:"h_bond"})
    adf = adf[adf["h_bond"]!=0.0]
    edges = torch.tensor(adf[["A","B"]].to_numpy())

    return edges

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