import os

from torch_geometric.datasets import TUDataset


def download_dataset(dataset_name: str, raw_dir: str = "data/raw"):
    """Download dataset using PyTorch Geometric's dataset utilities."""
    dataset_path = os.path.join(raw_dir, dataset_name)
    os.makedirs(raw_dir, exist_ok=True)
    dataset = TUDataset(root=dataset_path, name=dataset_name)
    print(f"Dataset {dataset_name} downloaded to {dataset_path}")
    return dataset


# Testing
if __name__ == "__main__":
    datasets = ["MUTAG", "IMDB-BINARY"]
    for dataset_name in datasets:
        download_dataset(dataset_name)
