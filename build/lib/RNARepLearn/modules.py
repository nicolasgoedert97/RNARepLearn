import torch
from torch.nn import LSTM, Conv1d, MultiheadAttention
import torch_geometric
from torch_geometric.nn import Linear, Sequential
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from .layers import RPINetGNNLayer



# H(l-1)W
class LinearEmbedding(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.lin = Linear(input_channels, output_channels).double()

    def forward(self, data):
        data.x = self.lin(data.x)
        return data

class RPINetEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, n_layers, conv_kernel_size ,dim_step=None):
        super().__init__()

        # LSTM --> cell memory stores info about each unrolling step of the GNN
        lstm = LSTM(output_channels, output_channels)

        layers = []
        final_output_channels = output_channels
        for i in range(n_layers):

            if dim_step is not None:
                output_channels = input_channels*dim_step if input_channels*dim_step <= final_output_channels else final_output_channels

            # add GNN layer
            layers.append( (RPINetGNNLayer(input_channels, output_channels, conv_kernel_size), 'x, edge_index, batch, edge_weight -> x' ))

            # add lstm call -> updated embeddings
            layers.append( (lstm, 'x -> x, (h, c)' ))

            input_channels = output_channels
            

        self.body = Sequential('x, edge_index, batch, edge_weight', layers)
            

    def forward(self, batch):

        batch.x, _ = self.body(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        
        return batch

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

class AttentionDecoderV2(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.nuc_projection = Linear(input_channels, output_channels)
        self.attention = MultiheadAttention(input_channels, 2)
        
    def forward(self, data):
        x = data.x
        
        x, attention_weights = self.attention(x,x,x)
        
        nucleotides = self.nuc_projection(x)
        
        return F.softmax(nucleotides, dim=1), F.softmax(attention_weights, dim=1)
