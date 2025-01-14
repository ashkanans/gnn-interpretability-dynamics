import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from models.base_model import BaseGNN


class GCN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=torch.nn.ReLU):
        """
        Graph Convolutional Network (GCN).
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
            num_layers (int): Number of GNN layers.
        """
        super(GCN, self).__init__(input_dim, hidden_dim, output_dim, num_layers, activation)

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc_out(x)
        return x
