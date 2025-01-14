import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from models.base_model import BaseGNN


class GIN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=torch.nn.ReLU):
        """
        Graph Isomorphism Network (GIN).
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
            num_layers (int): Number of GNN layers.
        """
        super(GIN, self).__init__(input_dim, hidden_dim, output_dim, num_layers, activation)

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GINConv(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation()
            )))

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc_out(x)
        return x
