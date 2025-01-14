import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv

from interpretability.concept_extraction import ConceptExtractor
from models.gin import GIN


def test_concept_extraction_framework():
    """
    Test the functionality of the ConceptExtractor framework.
    """
    # Prepare mock data
    dataset = TUDataset(root="data/raw/MUTAG", name="MUTAG")
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize a GNN model
    model = GIN(input_dim=7, hidden_dim=16, output_dim=2, num_layers=2)

    # Mock forward pass with random weights
    for data in data_loader:
        data = data.to("cpu")
        with torch.no_grad():
            activations = model(data.x, data.edge_index, data.batch)
        break  # Use a single batch for testing

    # Test base concept extraction
    labels = dataset[0].y  # Assuming labels are consistent across dataset
    base_concepts = [torch.ones(activations.size(0)), torch.zeros(activations.size(0))]
    assert len(base_concepts) > 0, "Base concepts should not be empty."
    assert isinstance(base_concepts[0], torch.Tensor), "Base concepts should be tensors."

    # Initialize ConceptExtractor
    extractor = ConceptExtractor(beam_width=min(10, len(base_concepts)), max_depth=3)

    # Test IoU computation
    concept_mask = (labels == labels[0]).float()
    iou_score = extractor.evaluate_concept(activations, concept_mask)
    assert 0.0 <= iou_score <= 1.0, "IoU score should be between 0 and 1."

    # Test compositional concept discovery
    best_concepts = extractor.beam_search(base_concepts, activations)
    assert len(best_concepts) > 0, "Beam search should return at least one concept."
    for concept, score in best_concepts:
        assert isinstance(concept, torch.Tensor), "Compositional concepts should be tensors."
        assert isinstance(score, float), "Scores should be floats."
        assert 0.0 <= score <= 1.0, "Concept scores should be between 0 and 1."

    # Test full concept extraction pipeline
    neuron_concepts = extractor.extract_concepts(model, data_loader)
    assert isinstance(neuron_concepts, dict), "Neuron concepts should be a dictionary."
    assert len(neuron_concepts) > 0, "Neuron concepts should not be empty."
    for neuron_idx, concepts in neuron_concepts.items():
        assert isinstance(neuron_idx, int), "Neuron indices should be integers."
        assert len(concepts) > 0, f"Neuron {neuron_idx} should have at least one concept."


# Mock GNN model for testing
class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv1 = GCNConv(3, 2)  # Input: 3 features, Output: 2 features

    def forward(self, x, edge_index, batch=None):
        return self.conv1(x, edge_index)


def test_concept_extractor():
    # Dummy activations and labels
    activations = torch.tensor([
        [0.2, 0.8, 0.1],
        [0.9, 0.7, 0.3],
        [0.4, 0.6, 0.8],
        [0.5, 0.9, 0.4]
    ])
    labels = torch.tensor([0, 1, 0, 1])

    # Prepare data
    data = Data(x=activations, edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]), y=labels)
    data_loader = [data]

    # Initialize the mock model and extractor
    model = MockModel()
    extractor = ConceptExtractor(beam_width=3, max_depth=2)

    # Test extract_base_concepts
    base_concepts = extractor.extract_base_concepts(activations, labels)
    print("Base Concepts:", base_concepts)
    assert len(base_concepts) > 0, "No base concepts extracted."

    # Test IoU computation
    iou_score = extractor.evaluate_concept(activations, base_concepts[0])
    print("IoU Score for first base concept:", iou_score)
    assert 0 <= iou_score <= 1, "IoU score out of valid range."

    # Test beam_search
    best_concepts = extractor.beam_search(base_concepts, activations)
    print("Best Concepts:", best_concepts)
    assert len(best_concepts) > 0, "No concepts found in beam search."

    # Test extract_concepts
    neuron_concepts = extractor.extract_concepts(model, data_loader, device="cpu")
    print("Neuron Concepts:", neuron_concepts)
    assert isinstance(neuron_concepts, dict), "Neuron concepts should be a dictionary."


if __name__ == "__main__":
    # Uncomment the test you want to run
    test_concept_extraction_framework()
    test_concept_extractor()
