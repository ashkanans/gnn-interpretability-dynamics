import torch


class GlobalExplanations:
    """
    Produces model-level explanations by identifying key neurons contributing to each class
    and aligning them with extracted concepts.
    """

    def __init__(self, top_k_neurons=3):
        """
        Args:
            top_k_neurons (int): Number of top neurons to consider for each class.
        """
        self.top_k_neurons = top_k_neurons

    def identify_key_neurons(self, model, data_loader, device="cpu"):
        """
        Identify the top-k neurons contributing to each class.
        Args:
            model (torch.nn.Module): Trained GNN model.
            data_loader (torch_geometric.data.DataLoader): DataLoader for input data.
            device (str): Device to run computation on ("cpu" or "cuda").
        Returns:
            dict: A dictionary mapping class indices to their top neurons.
        """
        model.to(device)
        model.eval()

        class_contributions = {}
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                outputs = model(data.x, data.edge_index, data.batch)  # Model output [num_graphs, num_classes]
                for class_idx in range(outputs.size(1)):  # Iterate over classes
                    class_contributions[class_idx] = class_contributions.get(class_idx, 0) + outputs[:, class_idx]

        # Identify top-k neurons for each class
        top_neurons = {}
        for class_idx, contributions in class_contributions.items():
            neuron_scores = contributions.mean(dim=0)  # Average contribution per neuron
            top_neurons[class_idx] = torch.topk(neuron_scores, self.top_k_neurons).indices.tolist()

        return top_neurons

    def align_neurons_with_concepts(self, activations, neuron_concepts, top_neurons):
        """
        Align key neurons with their extracted concepts.
        Args:
            activations (torch.Tensor): Neuron activations [num_nodes, num_neurons].
            neuron_concepts (dict): Extracted concepts for each neuron.
            top_neurons (dict): Key neurons for each class.
        Returns:
            dict: A mapping of classes to neurons and their associated concepts.
        """
        class_explanations = {}
        for class_idx, neurons in top_neurons.items():
            class_explanations[class_idx] = []
            for neuron_idx in neurons:
                concepts = neuron_concepts.get(neuron_idx, [])
                class_explanations[class_idx].append({
                    "neuron_idx": neuron_idx,
                    "concepts": concepts
                })
        return class_explanations

    def generate_logical_descriptions(self, class_explanations):
        """
        Generate logical descriptions for extracted concepts.
        Args:
            class_explanations (dict): Explanations per class with neurons and concepts.
        Returns:
            dict: Logical descriptions for each class.
        """
        logical_descriptions = {}
        for class_idx, neuron_info in class_explanations.items():
            descriptions = []
            for neuron in neuron_info:
                neuron_idx = neuron["neuron_idx"]
                concepts = neuron["concepts"]
                logical_description = " OR ".join(
                    [" AND ".join(map(str, concept[0].tolist())) for concept in concepts]
                )
                descriptions.append(f"Neuron {neuron_idx}: ({logical_description})")
            logical_descriptions[class_idx] = descriptions
        return logical_descriptions

    def explain(self, model, data_loader, neuron_concepts, activations, device="cpu"):
        """
        Produce global explanations for the model.
        Args:
            model (torch.nn.Module): Trained GNN model.
            data_loader (torch_geometric.data.DataLoader): DataLoader for input data.
            neuron_concepts (dict): Extracted concepts for each neuron.
            activations (torch.Tensor): Neuron activations [num_nodes, num_neurons].
            device (str): Device to run computation on ("cpu" or "cuda").
        Returns:
            dict: Logical explanations for each class.
        """
        # Identify key neurons for each class
        top_neurons = self.identify_key_neurons(model, data_loader, device)

        # Align key neurons with concepts
        class_explanations = self.align_neurons_with_concepts(activations, neuron_concepts, top_neurons)

        # Generate logical descriptions
        logical_descriptions = self.generate_logical_descriptions(class_explanations)

        return logical_descriptions
