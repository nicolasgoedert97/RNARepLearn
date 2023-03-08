from .modules import LinearEmbedding, RPINetEncoder, AttentionDecoder, TE_Decoder, CNN_Encoder, GCN_Encoder, CNN_Seq
from .layers import Sep_Seq_Struc_Layer
from .utils import add_backbone
import torch
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
    def __init__(self, input_channels, output_channels, N, struc_op, seq_op = CNN_Seq, act = F.relu, dropout=True, **kwargs):
        super().__init__()
        layers = []
        for i in range(N):
            layers.append( (Sep_Seq_Struc_Layer(input_channels, output_channels, seq_op, struc_op, **kwargs), 'x, edge_index, edge_weight, batch -> x') )
            layers.append( (act, 'x -> x') )
            input_channels = output_channels

            if dropout:
                layers.append((F.dropout, 'x -> x'))
        
        self.body = Sequential('x, edge_index, edge_weight, batch', layers)
    
    def forward(self, batch):

        x = self.body(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        batch.x = x

        return batch

@gin.register
class CombinedGNN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, op, **kwargs):
        super().__init__()
        layers = []
        for i in range(N):
            layers.append( (op(input_channels, output_channels, **kwargs), 'x, edge_index, edge_attr -> x') )
            input_channels = output_channels

        self.body = Sequential('x, edge_index, edge_attr', layers)
    
    def forward(self, batch):
        #batch = add_backbone(batch)

        x = self.body(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = x

        return batch