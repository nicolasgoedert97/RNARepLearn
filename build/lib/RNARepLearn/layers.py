import torch
import torch_geometric
from torch_geometric.nn import Linear, GCNConv
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

