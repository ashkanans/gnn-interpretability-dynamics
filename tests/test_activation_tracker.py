import os

import torch

from interpretability.activation_hooks import ActivationTracker
from models.gin import GIN


def test_activation_tracker():
    """
    Test the functionality of ActivationTracker.
    """
    # Initialize model and tracker
    model = GIN(input_dim=10, hidden_dim=32, output_dim=2, num_layers=3)
    tracker = ActivationTracker()

    # Register hooks
    tracker.register_hooks(model)

    # Simulate a forward pass
    x = torch.randn((50, 10))  # 50 nodes with 10 features each
    edge_index = torch.randint(0, 50, (2, 100))  # 100 edges
    batch = torch.zeros(50, dtype=torch.long)  # Single graph
    model(x, edge_index, batch)

    # Check that activations are captured
    assert len(tracker.activations) > 0, "No activations were captured."

    # Verify activations' structure
    for module, activation in tracker.activations.items():
        assert isinstance(activation, torch.Tensor), "Activations must be tensors."
        assert activation.size(0) > 0, "Activation tensor should not be empty."

    # Save activations
    file_path = "test_activations.pth"
    tracker.save_activations(file_path)
    assert os.path.exists(file_path), "Activations file was not saved."

    # Load activations and verify structure
    loaded_activations = torch.load(file_path)
    assert isinstance(loaded_activations, dict), "Loaded activations should be a dictionary."
    assert len(loaded_activations) == len(tracker.activations), "Loaded activations should match captured activations."

    # Clear activations and verify
    tracker.clear()
    assert len(tracker.activations) == 0, "Activations were not cleared."

    # Cleanup test file
    os.remove(file_path)
    assert not os.path.exists(file_path), "Test activation file was not removed."


if __name__ == "__main__":
    test_activation_tracker()
