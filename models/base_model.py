import torch.nn as nn


class BaseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU):
        """
        Base class for GNNs.
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output layer.
            num_layers (int): Number of GNN layers.
            activation (nn.Module): Activation function class (default: ReLU).
        """
        super(BaseGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()

    def forward(self, x, edge_index):
        raise NotImplementedError("Subclasses must implement the forward method.")
