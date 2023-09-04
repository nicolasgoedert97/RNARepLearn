import torch
import gin
from torch.nn import LSTM, Conv1d, MultiheadAttention, BatchNorm1d
import torch_geometric
from torch_geometric.nn import Linear, GCNConv, Sequential, global_mean_pool
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from .layers import RPINetGNNLayer, GCN_CNN_Layer



# H(l-1)W
class LinearEmbedding(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.lin = Linear(input_channels, output_channels).double()

    def forward(self, data):
        data.x = self.lin(data.x)
        return data


class GNN_Encoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, layer, n_layers ,dim_step=None):
        super().__init__()
        layers = []
        final_output_channels = output_channels
        for i in range(n_layers):

            if dim_step is not None:
                output_channels = input_channels*dim_step if input_channels*dim_step <= final_output_channels else final_output_channels

            # add GNN layer
            layers.append( (layer(input_channels, output_channels), 'x, edge_index, batch, edge_weight -> x' ))

            input_channels = output_channels
            
        self.body = Sequential('x, edge_index, batch, edge_weight', layers)


    def forward(self, batch):

        out, _ = self.body(batch.x, batch.edge_index, batch.batch, batch.edge_weight)

        out_batch = batch.clone()
        out_batch.x = out
        
        return out_batch


#@gin.configurable
@gin.register
class RPINetEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, n_layers, conv_kernel_size ,dim_step=None, struc_op = None):
        super().__init__()
        self.output_channels = output_channels
        
        layers = []
        final_output_channels = output_channels
        for i in range(n_layers):

            if dim_step is not None:
                output_channels = input_channels*dim_step if input_channels*dim_step <= final_output_channels else final_output_channels

            # add GNN layer
            layers.append( (RPINetGNNLayer(input_channels, output_channels, conv_kernel_size, struc_op), 'x, h0, c0, edge_index, batch, edge_weight -> x, batched_x, c0' ))

            input_channels = output_channels
            
        layers.append((LSTM(output_channels, output_channels, bidirectional=True), 'batched_x -> out, (h,c)'))
        self.body = Sequential('x, h0, c0, edge_index, batch, edge_weight', layers)
            

    def forward(self, batch):

        # for message passing step l= 1 cell memory = None --> initialize cell memory with zeros
        out, _ = self.body(batch.x.double(), None, None, batch.edge_index, batch.batch, batch.edge_weight)

        _, mask = to_dense_batch(batch.x, batch.batch)

        batch.x = out[mask]
        
        return batch

@gin.register
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
        
        return F.softmax(nucleotides, dim=1), F.log_softmax(dotprod, dim=1)

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

class Affinity_Decoder(torch.nn.Module):
    def __init__(self,input_channels, batch_size, n_prots, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.linear_in = torch.nn.Linear(input_channels, input_channels*2)
        self.linear_hidden = torch.nn.Linear(input_channels*2, input_channels*2)
        self.linear_out = torch.nn.Linear(input_channels*2, n_prots)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batch_size = batch_size
        
        self.lin_direct = torch.nn.Linear(n_prots, n_prots)
        
        #self.gat = GATConv(input_channels, n_prots)



    def forward(self, batch):
        x = batch.x

        x = self.linear_in(x)
        
        x = self.activation(x)
        
        x = self.linear_hidden(x)
        
        x = self.activation(x)
        
        x = self.linear_out(x)
        
        x = global_mean_pool(x, batch.batch)
        
        x = self.lin_direct(x)
        

        return x



@gin.configurable
class TE_Decoder(torch.nn.Module):
    def __init__(self, batch_size ,input_channels):
        super().__init__()
        self.batch_size = batch_size
        self.linear = Linear(input_channels, 1)
    
    def forward(self, batch):
        

        x = global_mean_pool(batch.x, batch.batch)

        x = self.linear(x)

        return x.squeeze()
        
        
@gin.register
class CNN_Encoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.conv1 = Conv1d(input_channels, output_channels, kernel_size,padding='same').double()
        self.conv2 = Conv1d(output_channels, output_channels, kernel_size,padding='same').double()

        self.bn1 = BatchNorm1d(output_channels).double()
        self.bn2 = BatchNorm1d(output_channels).double()
    
    def forward(self, batch):
        x_batched, fake_nodes_mask = to_dense_batch(batch.x, batch.batch)
        x_batched = torch.transpose(x_batched, 1, 2)

        x = self.conv1(x_batched)
        
        x = self.bn1(x)

        x = self.conv2(x)

        x = self.bn2(x)

        batch.x = torch.transpose(x, 1, 2)[fake_nodes_mask]
        
        return batch

class GCN_Encoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, n_layers):
        super().__init__()
        self.kernel_size = kernel_size

        layers = []
        for i in range(n_layers):
            layers.append((GCN_CNN_Layer(input_channels, output_channels, kernel_size), 'x, edge_index, edge_weight, batch -> x'))
            input_channels = output_channels
        
        self.body = Sequential('x, edge_index, edge_weight, batch', layers)

    def forward(self, batch):

        out = self.body(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        batch.x = out

        return batch


class CNN_Seq(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=9):
        super().__init__()
        self.body = Conv1d(input_channels, output_channels, kernel_size, padding='same')

    def forward(self, x, batch):
        x_batched, mask = to_dense_batch(x, batch)
        x_batched = torch.transpose(x_batched, 1, 2)

        x_batched = self.body(x_batched)




        return torch.transpose(x_batched, 1, 2)[mask]


class pooling_layer(torch.nn.Module):
    def __init__(self, pooling_method):
        super().__init__()
        self.pool = pooling_method

    def forward(self, batch):
        batch.x = self.pool(batch.x, batch.batch)

        return batch