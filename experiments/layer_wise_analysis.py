import torch
from torch_geometric.data import DataLoader

from interpretability.concept_extraction import ConceptExtractor
from interpretability.metrics import compute_iou, compute_absolute_contribution, compute_entropy


def analyze_layer_concepts(model, data_loader, device="cpu"):
    """
    Analyze layer-wise concepts in a GNN model.
    Args:
        model (torch.nn.Module): The GNN model to analyze.
        data_loader (DataLoader): DataLoader for the dataset.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        dict: Layer-wise analysis containing interpretability metrics and concept contributions.
    """
    model.to(device)
    model.eval()

    # Hook to capture activations at each layer
    layer_activations = []

    def hook_fn(module, input, output):
        layer_activations.append(output)

    # Register hooks on all GNN layers
    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    # Perform forward pass to capture activations
    data = next(iter(data_loader))  # Use a single batch for analysis
    data = data.to(device)
    with torch.no_grad():
        model(data.x, data.edge_index, data.batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Initialize ConceptExtractor
    extractor = ConceptExtractor()

    # Analyze concepts for each layer
    layer_results = {}
    for layer_idx, activations in enumerate(layer_activations):
        print(f"Analyzing Layer {layer_idx + 1}/{len(layer_activations)}")
        labels = data.y  # Use graph-level labels
        base_concepts = extractor.extract_base_concepts(activations, labels)

        # Compute interpretability metrics for each concept
        iou_scores = [compute_iou(activations, concept) for concept in base_concepts]
        abs_scores = [compute_absolute_contribution(activations, concept) for concept in base_concepts]
        ent_scores = [compute_entropy(concept) for concept in base_concepts]

        # Layer summary
        layer_results[layer_idx] = {
            "num_concepts": len(base_concepts),
            "iou_mean": torch.tensor(iou_scores).mean().item(),
            "abs_mean": torch.tensor(abs_scores).mean().item(),
            "ent_mean": torch.tensor(ent_scores).mean().item(),
        }

        print(f"Layer {layer_idx + 1}: Num Concepts: {layer_results[layer_idx]['num_concepts']}, "
              f"Mean IoU: {layer_results[layer_idx]['iou_mean']:.4f}, "
              f"Mean ABS: {layer_results[layer_idx]['abs_mean']:.4f}, "
              f"Mean ENT: {layer_results[layer_idx]['ent_mean']:.4f}")

    return layer_results


def main(dataset_name, save_path, num_layers, model_class, device="cpu"):
    """
    Perform layer-wise concept analysis on a GNN model.
    Args:
        dataset_name (str): Name of the dataset (e.g., "MUTAG", "IMDB-BINARY").
        save_path (str): Path to save analysis results.
        num_layers (int): Number of layers in the GNN model.
        model_class: The GNN model class to analyze.
        device (str): Device to run computation on ("cpu" or "cuda").
    """
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    # Load dataset
    dataset = TUDataset(root=f"data/raw/{dataset_name}", name=dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = model_class(input_dim=num_features, hidden_dim=16, output_dim=num_classes, num_layers=num_layers)

    # Analyze concepts at each layer
    results = analyze_layer_concepts(model, data_loader, device)

    # Save results
    save_file = f"{save_path}/layer_analysis_{dataset_name}.pt"
    torch.save(results, save_file)
    print(f"Layer-wise analysis results saved to {save_file}")


if __name__ == "__main__":
    import argparse
    from models.gin import GIN  # Import your GIN implementation

    parser = argparse.ArgumentParser(description="Perform layer-wise concept analysis.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MUTAG, IMDB-BINARY).")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save analysis results.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the GNN model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation on ('cpu' or 'cuda').")

    args = parser.parse_args()

    main(args.dataset, args.save_path, args.num_layers, GIN, args.device)
