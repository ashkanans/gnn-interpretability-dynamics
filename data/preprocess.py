import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset


def preprocess_dataset(dataset_name: str, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Preprocess the dataset: split into train/validation/test sets."""
    raw_path = os.path.join(raw_dir, dataset_name)
    processed_path = os.path.join(processed_dir, dataset_name)
    os.makedirs(processed_path, exist_ok=True)

    dataset = TUDataset(root=raw_path, name=dataset_name)

    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split, split_indices in splits.items():
        split_data = [dataset[i] for i in split_indices]
        torch.save(split_data, os.path.join(processed_path, f"{split}.pt"))
        print(f"Saved {split} set with {len(split_data)} graphs to {processed_path}")


# Testing
if __name__ == "__main__":
    datasets = ["MUTAG", "IMDB-BINARY"]
    for dataset_name in datasets:
        preprocess_dataset(dataset_name)
