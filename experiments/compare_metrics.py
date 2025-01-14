import torch

from interpretability.concept_extraction import ConceptExtractor
from interpretability.metrics import compute_iou, compute_absolute_contribution, compute_entropy
from utils.plot_helpers import visualize_comparison


def evaluate_xgnn_baseline(model, data_loader, device="cpu"):
    """
    Simulate an XGNN-like baseline interpretability method.
    Args:
        model (torch.nn.Module): The trained GNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        dict: Simulated XGNN interpretability scores.
    """
    model.to(device)
    model.eval()

    xgnn_scores = {"ease_of_explanation": [], "intuition_alignment": []}
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)

            # Simulate interpretability scores (mock values for comparison)
            for _ in preds:
                xgnn_scores["ease_of_explanation"].append(torch.rand(1).item())
                xgnn_scores["intuition_alignment"].append(torch.rand(1).item())

    # Average scores
    xgnn_scores = {k: torch.tensor(v).mean().item() for k, v in xgnn_scores.items()}
    return xgnn_scores


def benchmark_metrics(model, data_loader, device="cpu"):
    """
    Benchmark interpretability metrics against XGNN.
    Args:
        model (torch.nn.Module): The trained GNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        dict: Comparison results for interpretability metrics.
    """
    extractor = ConceptExtractor()
    model.to(device)
    model.eval()

    # Extract interpretability metrics
    metric_scores = {"iou": [], "abs": [], "ent": []}
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            activations = model(data.x, data.edge_index, data.batch)

        labels = data.y
        base_concepts = extractor.extract_base_concepts(activations, labels)

        for concept in base_concepts:
            metric_scores["iou"].append(compute_iou(activations, concept))
            metric_scores["abs"].append(compute_absolute_contribution(activations, concept))
            metric_scores["ent"].append(compute_entropy(concept))

    # Average scores
    metric_scores = {k: torch.tensor(v).mean().item() for k, v in metric_scores.items()}

    # Simulate XGNN baseline
    xgnn_scores = evaluate_xgnn_baseline(model, data_loader, device)

    return {"our_metrics": metric_scores, "xgnn_baseline": xgnn_scores}


def main(dataset_name, save_path, model_class, num_layers, device="cpu"):
    """
    Compare interpretability metrics for a given dataset and model.
    Args:
        dataset_name (str): Name of the dataset (e.g., MUTAG, IMDB-BINARY).
        save_path (str): Path to save results and visualizations.
        model_class: GNN model class to analyze.
        num_layers (int): Number of layers in the GNN model.
        device (str): Device to run computation on ("cpu" or "cuda").
    """
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    # Load dataset
    dataset = TUDataset(root=f"data/raw/{dataset_name}", name=dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data_loader = DataLoader(dataset, batch_size=16)

    # Initialize GNN model
    model = model_class(input_dim=num_features, hidden_dim=16, output_dim=num_classes, num_layers=num_layers)

    # Benchmark interpretability metrics
    results = benchmark_metrics(model, data_loader, device)

    # Visualize results
    visualize_comparison(results, save_path)

    # Save results
    torch.save(results, f"{save_path}/metric_comparison_{dataset_name}.pt")
    print(f"Comparison results saved to {save_path}/metric_comparison_{dataset_name}.pt")


if __name__ == "__main__":
    import argparse
    from models.gin import GIN  # Import your GIN implementation

    parser = argparse.ArgumentParser(description="Compare interpretability metrics for GNNs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MUTAG, IMDB-BINARY).")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save metrics and plots.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the GNN model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation on ('cpu' or 'cuda').")

    args = parser.parse_args()

    main(args.dataset, args.save_path, GIN, args.num_layers, args.device)
