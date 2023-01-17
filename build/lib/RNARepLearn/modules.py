import torch
import torch_geometric
from torch_geometric.nn import Linear, GCNConv
import torch.nn.functional as F
from .layers import LinearAggregation


# H(l-1)W
class LinearEmbedding(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.lin = Linear(input_channels, output_channels).double()

    def forward(self, data):
        data.x = self.lin(data.x)
        return data

class RPINetEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, n_layers):
        super().__init__()
        self.n_layers = n_layers
        
        #Conv(Covalent) + Aggr_H_bond (e.g. AH(l-1)W equation 2 from RPINet paper) TODO How to realise with/out fixed size of learnable matrice W?
        self.conv = GCNConv(input_channels, output_channels)
        self.conv1d = torch.nn.Conv1d(input_channels, output_channels, 3)
        self.lin = LinearAggregation(input_channels, output_channels)



    def forward(self, batch):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_weight

        cov_bonds=[]

        batch_offset = 0
        for data in batch.to_data_list():
            for tupl in [(int(i),int(i+1)) for i in range(batch_offset,data.x.shape[0]-1+batch_offset)]:
                        cov_bonds.append(tupl)
                        cov_bonds.append((tupl[1],tupl[0]))
            batch_offset+=data.x.shape[0]

        cov_bonds = torch.tensor(cov_bonds).T.cuda()


        for H in range(self.n_layers-1):
            #Linear transform of nodes -> AH(l-1)W
            #lin_x = torch.matmul(torch.sparse_coo_tensor(data.edge_index, data.edge_weight).to_dense(), x) SVM alternative
            #lin_x = self.lin(x, edge_index, edge_weight)
            
            
            #Convolution on covalent bonds
            #conv_x_covalent = self.lin(x, cov_bonds, None)

            conv_x_h_bonds = self.lin(x, edge_index, edge_weight)
            
            x = conv_x_h_bonds#lin_x + conv_x_covalent

            x = F.relu(x)

            F.dropout(x, training=self.training)


        data.x = x
        
        return data

class AttentionDecoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.key_projection = Linear(input_channels, input_channels)
        self.query_projection = Linear(input_channels, input_channels)
        self.nuc_projection = Linear(input_channels, output_channels)
        
    def forward(self, data):
        x = data.x
        keys = self.key_projection(x)
        queries = self.query_projection(x)
        
        nucleotides = self.nuc_projection(x)
        
        dotprod = torch.matmul(queries,keys.T)
        
        return F.softmax(nucleotides, dim=1), F.softmax(dotprod, dim=1)