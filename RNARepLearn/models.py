from .modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder, CNN_Encoder, GCN_Encoder, CNN_Seq
from .layers import Sep_Seq_Struc_Layer, Sep_Seq_Struc_Layer_LSTM
from .utils import add_backbone
import torch
from torch.nn import LSTM
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import Sequential, TransformerConv
import gin
import torch.nn.functional as F

@gin.configurable
class Encoder_Decoder_Model(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        layers = []

        if isinstance(encoder, torch.nn.Module):
            layers.append(encoder)
        else:
            layers.append(encoder())

        if isinstance(decoder, torch.nn.Module):
            layers.append(decoder)
        else:
            layers.append(decoder())

        self.model = torch.nn.Sequential(*layers).double()
    
    def forward(self, batch):
        

        return self.model(batch)
        

@gin.register
class Seq_Struc_GNN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, struc_op, seq_op = CNN_Seq, act = F.relu, dropout=True, to_float32=False, **kwargs):
        super().__init__()
        self.input_cahnnels = input_channels
        self.output_channels = output_channels
        layers = []
        for i in range(N):
            layers.append( (Sep_Seq_Struc_Layer(input_channels, output_channels, seq_op, struc_op, **kwargs), 'x, edge_index, edge_weight, batch -> x') )
            layers.append( (act, 'x -> x') )
            input_channels = output_channels

            if dropout:
                layers.append((F.dropout, 'x -> x'))
        
        self.body = Sequential('x, edge_index, edge_weight, batch', layers)

        self.to_float32=to_float32
    
    def forward(self, batch):

        if self.to_float32:
            batch.x = batch.x.to(torch.float32)
            batch.edge_weight = batch.edge_weight.to(torch.float32)

        x = self.body(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        batch.x = x

        return batch

@gin.register
class Seq_Struc_GNN_LSTM(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, struc_op, seq_op = CNN_Seq, dropout=True, to_float32=False, **kwargs):
        super().__init__()
        self.input_cahnnels = input_channels
        self.output_channels = output_channels
        layers = []
        for i in range(N):
            layers.append( (Sep_Seq_Struc_Layer_LSTM(input_channels, output_channels, seq_op, struc_op, **kwargs), 'x, edge_index, edge_weight, batch, cell_mem -> x, batched_x ,cell_mem') )
            input_channels = output_channels

            if dropout:
                layers.append((F.dropout, 'x -> x'))
        
        layers.append((LSTM(output_channels, output_channels, bidirectional=True), 'batched_x -> out, (h,c)'))
        self.body = Sequential('x, edge_index, edge_weight, batch, cell_mem', layers)

        self.to_float32=to_float32
    
    def forward(self, batch):

        if self.to_float32:
            batch.x = batch.x.to(torch.float32)
            batch.edge_weight = batch.edge_weight.to(torch.float32)

        out, _ = self.body(batch.x, batch.edge_index, batch.edge_weight, batch.batch, None)

        _, mask = to_dense_batch(batch.x, batch.batch)

        batch.x = out[mask]

        return batch

@gin.register
class CombinedGNN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, op, **kwargs):
        super().__init__()
        layers = []
        for i in range(N):
            layers.append( (op(input_channels, output_channels, **kwargs), 'x, edge_index, edge_weight -> x') )
            input_channels = output_channels

        self.body = Sequential('x, edge_index, edge_weight', layers)
    
    def forward(self, batch):
        #batch = add_backbone(batch)

        x = self.body(batch.x, batch.edge_index, batch.edge_weight)

        batch.x = x

        return batch

