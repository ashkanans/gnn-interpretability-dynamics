import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from models.gin import GIN  # Import your GIN implementation


def evaluate_model(model, data_loader, device="cpu"):
    """
    Evaluate the accuracy of a trained GNN model.
    Args:
        model (torch.nn.Module): The trained GNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        float: Model accuracy.
    """
    model.to(device)
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


def benchmark_accuracy(datasets, model_class, num_layers, hidden_dim, batch_size, epochs, save_path, device="cpu"):
    """
    Benchmark the accuracy of GNN models across datasets and training setups.
    Args:
        datasets (list): List of dataset names to benchmark.
        model_class: GNN model class to benchmark.
        num_layers (int): Number of layers in the GNN model.
        hidden_dim (int): Hidden dimension size for the GNN model.
        batch_size (int): Batch size for training and evaluation.
        epochs (int): Number of training epochs.
        save_path (str): Path to save results and plots.
        device (str): Device to run computation on ("cpu" or "cuda").
    """
    results = []
    os.makedirs(save_path, exist_ok=True)

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")

        # Load dataset
        dataset = TUDataset(root=f"data/raw/{dataset_name}", name=dataset_name)
        num_features = dataset.num_features
        num_classes = dataset.num_classes

        train_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], dataset[int(0.8 * len(dataset)):]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model
        model = model_class(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes,
                            num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train model
        model.to(device)
        model.train()
        for epoch in range(1, epochs + 1):
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = torch.nn.functional.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()

        # Evaluate model
        accuracy = evaluate_model(model, test_loader, device)
        results.append({"Dataset": dataset_name, "Accuracy": accuracy})
        print(f"Dataset: {dataset_name}, Accuracy: {accuracy:.4f}")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results as a CSV file
    results_file = os.path.join(save_path, "accuracy_benchmark.csv")
    results_df.to_csv(results_file, index=False)

    # Plot results
    plot_file = os.path.join(save_path, "accuracy_benchmark.png")
    results_df.plot(x="Dataset", y="Accuracy", kind="bar", legend=False, color="skyblue")
    plt.title("GNN Model Accuracy Benchmark")
    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()

    print(f"Benchmark results saved to {results_file}")
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GNN model accuracy across datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="List of datasets (e.g., MUTAG IMDB-BINARY).")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save benchmark results and plots.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the GNN model.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension size for the GNN model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation on ('cpu' or 'cuda').")

    args = parser.parse_args()

    benchmark_accuracy(
        datasets=args.datasets,
        model_class=GIN,  # Assuming GIN is the default model for benchmarking
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_path=args.save_path,
        device=args.device,
    )
