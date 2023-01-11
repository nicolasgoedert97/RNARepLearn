import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCNConv_layer(torch.nn.Module):
    def __init__(self, node_features, output_dimension,dropout_rate=0.02):
        super().__init__()
        self.conv1 = GCNConv(node_features, output_dimension)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        #build representation
        # input -> V_N,D
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return F.softmax(x, dim=1)
