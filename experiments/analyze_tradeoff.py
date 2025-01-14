import os

import torch
from sklearn.metrics import accuracy_score

from interpretability.concept_extraction import ConceptExtractor
from interpretability.metrics import compute_iou, compute_absolute_contribution, compute_entropy
from models.gin import GIN  # Import your GIN model implementation
from utils.plot_helpers import plot_tradeoff


def evaluate_model(model, data_loader, device="cpu"):
    """
    Evaluate model accuracy on a dataset.
    Args:
        model (torch.nn.Module): Trained GNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        float: Model accuracy.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return accuracy_score(all_labels, all_preds)


def analyze_tradeoff(model, train_loader, test_loader, dataset_name, save_path, num_epochs=50, device="cpu"):
    """
    Analyze interpretability-accuracy trade-off over training epochs.
    Args:
        model (torch.nn.Module): GNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        dataset_name (str): Name of the dataset.
        save_path (str): Path to save results and visualizations.
        num_epochs (int): Number of training epochs.
        device (str): Device to run computation on ("cpu" or "cuda").
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    extractor = ConceptExtractor()

    metrics = {"epochs": [], "accuracy": [], "iou": [], "abs": [], "ent": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

        # Evaluate accuracy
        accuracy = evaluate_model(model, test_loader, device)

        # Extract interpretability metrics
        data = next(iter(test_loader))  # Use the first batch for interpretability
        data = data.to(device)
        activations = model(data.x, data.edge_index, data.batch)
        labels = data.y
        base_concepts = extractor.extract_base_concepts(activations, labels)

        iou_score = sum(compute_iou(activations, concept) for concept in base_concepts) / len(base_concepts)
        abs_score = sum(compute_absolute_contribution(activations, concept) for concept in base_concepts) / len(
            base_concepts)
        ent_score = sum(compute_entropy(concept) for concept in base_concepts) / len(base_concepts)

        # Save metrics
        metrics["epochs"].append(epoch)
        metrics["accuracy"].append(accuracy)
        metrics["iou"].append(iou_score)
        metrics["abs"].append(abs_score)
        metrics["ent"].append(ent_score)

        print(
            f"Epoch {epoch}/{num_epochs}: Accuracy: {accuracy:.4f}, IoU: {iou_score:.4f}, ABS: {abs_score:.4f}, ENT: {ent_score:.4f}")

    # Save metrics to file
    torch.save(metrics, os.path.join(save_path, f"metrics_tradeoff_{dataset_name}.pt"))

    # Plot interpretability-accuracy trade-off
    plot_tradeoff(metrics, save_path, dataset_name)


if __name__ == "__main__":
    import argparse
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    parser = argparse.ArgumentParser(description="Analyze interpretability-accuracy trade-off.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MUTAG, IMDB-BINARY).")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save metrics and plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension size for the GNN model.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the GNN model.")

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Load dataset
    dataset = TUDataset(root=f"data/raw/{args.dataset}", name=args.dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    train_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize GNN model
    model = GIN(input_dim=num_features, hidden_dim=args.hidden_dim, output_dim=num_classes,
                num_layers=args.num_layers).to("cpu")

    # Run analysis
    analyze_tradeoff(model, train_loader, test_loader, args.dataset, args.save_path, args.epochs)
