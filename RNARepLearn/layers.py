import torch
from torch.nn import LSTM, Conv1d
import torch_geometric
from torch_geometric.nn import Linear, GCNConv
from torch_geometric.utils import to_dense_batch, remove_self_loops
import torch.nn.functional as F


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
    def __init__(self, input_channels, output_channels, filter_size):
        super().__init__()
        self.lin = Linear(input_channels, output_channels)
        self.conv = Conv1d(input_channels, output_channels, filter_size, padding='same')

    def forward(self,x, edge_index, batch, edge_weight):

        # remove self loops, such that aggregation happens only on neighbors. 
        edge_index, edge_weight = remove_self_loops(edge_index,edge_weight)

        # AH(l-1)W -- Linear projection H(l-1)W -> Aggregated over adjacent neighbour nodes (e.g. *A)
        lin_x = self.lin(x)
        lin_x = self.propagate(edge_index, x=lin_x, norm=edge_weight)

        # Convolution of filer_size'd filter over RNA backbone (size=3 is desirable to keep GNN logic, e.g. message from direct neighbors only)
        ## py_geometric databatch -> batched tensor / Adds fake nodes to keep length equal
        batched_x, fake_nodes_mask = to_dense_batch(x, batch)
        ## convolution 
        # TODO Check masking for correctness
        conv_x = self.conv(torch.transpose(batched_x, 1, 2))
        ## removal of fake nodes
        conv_x = torch.transpose(conv_x, 1, 2)[fake_nodes_mask]

        # As self loops are removed, linear = aggregated(add) messages of base pairings + conv(filter size 3) = messages from backbone neighbors
        return F.relu(lin_x+conv_x)

    def message(self, x_j, edge_index, norm):
        # normalize neighbor messages x_j by bpp of nodes
        return norm.view(-1,1) * x_j
