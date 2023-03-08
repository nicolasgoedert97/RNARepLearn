import torch
from torch.nn import LSTM, Conv1d
import torch_geometric
from torch_geometric.nn import Linear, GCNConv
from torch_geometric.utils import to_dense_batch, remove_self_loops
import torch.nn.functional as F
import gin


# H(l-1)AW
class LinearAggregation(torch_geometric.nn.MessagePassing):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lin = Linear(input_channels, output_channels).double() #H(l-1)W

    def forward(self,x, edge_index, edge_weight):
        x = self.lin(x)

        x = self.propagate(edge_index, x=x, norm=edge_weight)

        return x

    def message(self, x_j, edge_index, norm):
        return norm.view(-1,1) * x_j

class RPINetGNNLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, input_channels, output_channels, filter_size, struc_op=None):
        super().__init__()
        self.lin = Linear(input_channels, output_channels)
        self.conv = Conv1d(input_channels, output_channels, filter_size, padding='same')
        self.lstm = LSTM(output_channels, output_channels, batch_first=True)
        if struc_op is None:
            self.struc_op = struc_op
        else:
            self.struc_op = struc_op(input_channels, output_channels)

    def forward(self, x, h0, c0, edge_index, batch, edge_weight):
        #x = [b_S, Lenghth, embed]
        #h0 = [1,b_S, embed_dim]
        #
        
        ## py_geometric databatch -> batched tensor / Adds fake nodes to keep length equal
        batched_x, mask = to_dense_batch(x, batch)
        batched_x = torch.transpose(batched_x, 1, 2)



        ## convolution 
        conv_x = self.conv(batched_x)

        # AH(l-1)W -- Linear projection H(l-1)W -> Aggregated over adjacent neighbour nodes
        if self.struc_op is None:
            lin_x = self.lin(x)
            lin_x, _= to_dense_batch(self.propagate(edge_index, x=lin_x, norm=edge_weight), batch)

        else:
            lin_x = self.struc_op(x, edge_index, edge_weight)
            lin_x, _= to_dense_batch(lin_x, batch)
        
        # linear = aggregated(add) messages of base pairings + conv(filter size 3) = messages from backbone neighbors

        messages = torch.transpose(F.relu(torch.transpose(lin_x, 1, 2)+conv_x), 1,2)

        batched_x = torch.transpose(batched_x, 1, 2)

                # if l=1, set cell memory to
        if c0 is None:
            c0 = torch.zeros(1, batched_x.shape[0], messages.shape[2]).double()#.cuda()
        if h0 is None:
            h0 = torch.zeros(1, batched_x.shape[0], messages.shape[2]).double()#.cuda()

        
        output, (hn,cn) = self.lstm(messages, (h0, c0))


        #apply mask to 'unbatch' output tensor again [batch_size, x, y] -> [X,Y] combined graph
        return output[mask], output, cn

    def message(self, x_j, edge_index, norm):
        # normalize neighbor messages x_j by bpp of nodes
        return norm.view(-1,1) * x_j

class GCN_CNN_Layer(torch_geometric.nn.MessagePassing):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        self.bb_conv = Conv1d(input_channels, output_channels, padding='same', kernel_size=kernel_size)
        self.bp_conv = GCNConv(input_channels, output_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        dense_batch, mask = to_dense_batch(x,batch)
        bb = self.bb_conv(torch.transpose(dense_batch, 1, 2)).transpose(1,2)[mask]

        bp = self.bp_conv(x, edge_index, edge_weight)

        out = F.relu(bb+bp)

        out = F.dropout(out, training=self.training)

        return(out)

@gin.configurable  
class Sep_Seq_Struc_Layer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, seq_op, struc_op,gatedGraphConv=False, **kwargs):
        super().__init__()
        self.seq_op = seq_op(input_channels, output_channels)
        self.struc_op = struc_op(input_channels, output_channels)

        if gatedGraphConv:
            self.struc_op = struc_op(output_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        seq_x = self.seq_op(x, batch)
        struc_x = self.struc_op(x, edge_index, edge_weight)

        out = F.relu(seq_x+struc_x)

        return(out)