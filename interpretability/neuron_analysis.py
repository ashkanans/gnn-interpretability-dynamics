import torch
import torch.nn.functional as F


class NeuronAnalysis:
    """
    Tools to analyze neuron importance and detect redundancy in GNN models.
    """

    def __init__(self, model, data_loader, device="cpu"):
        """
        Initialize the NeuronAnalysis tool.
        Args:
            model (torch.nn.Module): The trained GNN model.
            data_loader (torch_geometric.data.DataLoader): DataLoader for input data.
            device (str): Device for computation ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def compute_neuron_importance(self):
        """
        Compute neuron importance scores based on their contributions to predictions.
        Returns:
            torch.Tensor: Importance scores for each neuron in the output layer.
        """
        self.model.eval()
        total_contributions = torch.zeros(self.model.fc_out.out_features).to(self.device)

        with torch.no_grad():
            for data in self.data_loader:
                data = data.to(self.device)
                activations = self.model(data.x, data.edge_index, data.batch)
                contributions = activations.abs().sum(dim=0)  # Sum absolute activations per neuron
                total_contributions += contributions

        importance_scores = total_contributions / total_contributions.sum()  # Normalize scores
        return importance_scores

    def detect_redundant_neurons(self, threshold=0.01):
        """
        Detect redundant neurons based on low importance or similar activation patterns.
        Args:
            threshold (float): Importance score threshold for pruning.
        Returns:
            dict: Dictionary containing redundant neurons based on importance and similarity.
        """
        # Compute neuron importance scores
        importance_scores = self.compute_neuron_importance()

        # Detect low-importance neurons
        low_importance_neurons = (importance_scores < threshold).nonzero(as_tuple=True)[0].tolist()

        # Analyze neuron activation similarity
        activations_list = []
        for data in self.data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                activations = self.model(data.x, data.edge_index, data.batch)
                activations_list.append(activations)

        # Combine activations across all batches
        all_activations = torch.cat(activations_list, dim=0)  # Shape: [num_nodes, num_neurons]
        similarity_matrix = F.cosine_similarity(all_activations.T.unsqueeze(1), all_activations.T.unsqueeze(0), dim=-1)
        redundant_pairs = (similarity_matrix > 0.95).nonzero(as_tuple=False).tolist()

        # Filter redundant pairs
        redundant_pairs = [(i, j) for i, j in redundant_pairs if i != j and i < j]

        return {
            "low_importance_neurons": low_importance_neurons,
            "redundant_pairs": redundant_pairs
        }

    def prune_neurons(self, redundant_neurons):
        """
        Prune redundant neurons from the model.
        Args:
            redundant_neurons (list): List of neuron indices to prune.
        Returns:
            torch.nn.Module: A pruned version of the model.
        """
        # Modify the output layer to exclude redundant neurons
        input_dim = self.model.fc_out.in_features
        output_dim = self.model.fc_out.out_features - len(redundant_neurons)

        # Create a new output layer
        new_fc_out = torch.nn.Linear(input_dim, output_dim, bias=True)
        retained_indices = [i for i in range(self.model.fc_out.out_features) if i not in redundant_neurons]

        # Copy weights and biases from the original layer
        new_fc_out.weight.data = self.model.fc_out.weight.data[retained_indices, :]
        new_fc_out.bias.data = self.model.fc_out.bias.data[retained_indices]

        # Replace the model's output layer
        self.model.fc_out = new_fc_out
        return self.model
