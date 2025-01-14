from torch import nn
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool

from models.base_model import BaseGNN


class GIN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GINConv(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )))
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.fc_out(x)
