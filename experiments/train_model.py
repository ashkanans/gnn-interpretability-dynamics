import os

import torch
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from models.gin import GIN


def train(model, optimizer, data_loader, device):
    """
    Train the GNN model for one epoch.
    Args:
        model (torch.nn.Module): The GNN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        data_loader (DataLoader): DataLoader for training data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # Model forward pass
        loss = cross_entropy(out, data.y)  # Compute cross-entropy loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs  # Accumulate loss

    return total_loss / len(data_loader.dataset)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate the GNN model on validation or test data.
    Args:
        model (torch.nn.Module): The GNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computation on ("cpu" or "cuda").
    Returns:
        float, float: Accuracy and average loss.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy, total_loss / len(data_loader.dataset)


def main(dataset_name, model_save_path, num_epochs=50, batch_size=32, learning_rate=0.01, hidden_dim=16, num_layers=3):
    """
    Train and evaluate a GNN model on a dataset.
    Args:
        dataset_name (str): Name of the dataset (e.g., "MUTAG", "IMDB-BINARY").
        model_save_path (str): Path to save the trained model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        hidden_dim (int): Hidden dimension size for the model.
        num_layers (int): Number of layers in the GNN model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = TUDataset(root=f"data/raw/{dataset_name}", name=dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    train_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    model = GIN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, num_layers=num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_accuracy = 0
    metrics = {"train_loss": [], "test_loss": [], "test_accuracy": []}
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        test_accuracy, test_loss = evaluate(model, test_loader, device)

        # Save metrics
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_loss)
        metrics["test_accuracy"].append(test_accuracy)

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, f"best_model_{dataset_name}.pt"))

        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save metrics
    torch.save(metrics, os.path.join(model_save_path, f"metrics_{dataset_name}.pt"))
    print(f"Training complete. Best Test Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MUTAG, IMDB-BINARY).")
    parser.add_argument("--save_path", type=str, default="models", help="Path to save the trained model and metrics.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension size for the GNN model.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the GNN model.")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args.dataset, args.save_path, args.epochs, args.batch_size, args.learning_rate, args.hidden_dim,
         args.num_layers)
