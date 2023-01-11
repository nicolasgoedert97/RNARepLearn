import random
import torch
import numpy as np
from scipy import sparse


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
        


