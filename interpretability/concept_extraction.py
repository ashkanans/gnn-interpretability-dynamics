import itertools

import torch

from interpretability.metrics import compute_entropy, compute_iou, compute_absolute_contribution


class ConceptExtractor:
    """
    Extracts concepts from GNN neurons using activation patterns.
    Includes base and compositional concepts with a beam search mechanism.
    """

    def __init__(self, beam_width=10, max_depth=3):
        """
        Args:
            beam_width (int): Maximum number of concepts to keep at each step.
            max_depth (int): Maximum depth of compositional concepts.
        """
        self.beam_width = beam_width
        self.max_depth = max_depth

    def extract_base_concepts(self, activations, labels):
        """
        Extract diverse base concepts from neuron activations and labels.
        Args:
            activations (torch.Tensor): Neuron activations [num_nodes, num_neurons].
            labels (torch.Tensor): Node labels or properties [num_nodes].
        Returns:
            list: Base concepts as diverse boolean masks.
        """
        base_concepts = []

        # Label-based concepts
        for label in torch.unique(labels):
            mask = (labels == label).float()
            base_concepts.append(mask)

        # Activation-based concepts
        activation_thresholds = torch.linspace(activations.min(), activations.max(), steps=3)
        for threshold in activation_thresholds:
            mask = (activations.mean(dim=1) > threshold).float()
            base_concepts.append(mask)

        # Filter trivial concepts
        base_concepts = [concept for concept in base_concepts if
                         not torch.all(concept == 1) and not torch.all(concept == 0)]

        return base_concepts

    def evaluate_concept(self, activations, concept):
        """
        Compute concept alignment score using Intersection over Union (IoU).
        Args:
            activations (torch.Tensor): Neuron activations [num_nodes, num_neurons].
            concept (torch.Tensor): Boolean mask for concept [num_nodes].
        Returns:
            float: IoU score for the concept.
        """
        # Apply threshold to activations
        threshold = torch.quantile(activations, 0.75, dim=0, keepdim=True)
        thresholded = (activations > threshold).float()

        # Compute intersection and union
        intersection = (thresholded * concept.unsqueeze(1)).sum(dim=0)
        union = (thresholded + concept.unsqueeze(1) > 0).float().sum(dim=0)

        # Compute IoU and return the mean score
        return (intersection / union).mean().item()

    def beam_search(self, base_concepts, activations):
        current_beam = base_concepts
        best_concepts = []
        seen_concepts = set()

        for depth in range(self.max_depth):
            candidate_concepts = []
            for concept_a, concept_b in itertools.combinations(current_beam, 2):
                for op in [torch.logical_and, torch.logical_or]:
                    candidate = op(concept_a, concept_b).float()

                    # Skip trivial and duplicate concepts
                    candidate_key = tuple(candidate.tolist())
                    if candidate_key in seen_concepts or torch.all(candidate == 1) or torch.all(candidate == 0):
                        continue
                    seen_concepts.add(candidate_key)

                    iou_score = self.evaluate_concept(activations, candidate)
                    print(f"Candidate: {candidate}, IoU Score: {iou_score}")  # Debug print
                    candidate_concepts.append((candidate, iou_score))

            # Keep top beam_width concepts
            candidate_concepts = sorted(candidate_concepts, key=lambda x: x[1], reverse=True)
            print(f"Candidate Concepts: {candidate_concepts}")  # Debug print
            current_beam = [c[0] for c in candidate_concepts[:self.beam_width]]
            best_concepts.extend(candidate_concepts[:self.beam_width])

        print(f"Best Concepts: {best_concepts}")  # Debug print
        return best_concepts

    def extract_concepts(self, model, data_loader, device="cpu"):
        """
        Extract concepts for all neurons in a GNN.
        Args:
            model (torch.nn.Module): Trained GNN model.
            data_loader (torch_geometric.data.DataLoader): DataLoader for input data.
            device (str): Device to run computation on ("cpu" or "cuda").
        Returns:
            dict: Concepts for each neuron and their IoU scores.
        """
        model.to(device)
        model.eval()

        all_activations, all_labels = [], []
        for data in data_loader:
            data = data.to(device)
            with torch.no_grad():
                activations = model(data.x, data.edge_index, data.batch)
            all_activations.append(activations)
            all_labels.append(data.y)

        activations = torch.cat(all_activations, dim=0)
        labels = torch.cat(all_labels, dim=0)

        base_concepts = self.extract_base_concepts(activations, labels)
        neuron_concepts = {}
        for neuron_idx in range(activations.size(1)):
            neuron_activations = activations[:, neuron_idx]
            best_concepts = self.beam_search(base_concepts, neuron_activations)
            neuron_concepts[neuron_idx] = best_concepts

        return neuron_concepts

    def compute_iou(self, activations, concept):
        return compute_iou(activations, concept)

    def compute_absolute_contribution(self, activations, concept):
        return compute_absolute_contribution(activations, concept)

    def compute_entropy(self, concept):
        return compute_entropy(concept)
