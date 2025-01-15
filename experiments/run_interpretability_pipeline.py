import os

import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from interpretability.concept_extraction import ConceptExtractor
from models.gin import GIN
from utils.plot_helpers import visualize_graph_concept_activations, visualize_chemical_graph


def main():
    # Step 1: Load the MUTAG dataset
    dataset_name = "MUTAG"
    dataset = TUDataset(root=f"../data/raw/{dataset_name}", name=dataset_name)
    train_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], dataset[int(0.8 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Step 2: Define and Train the GNN model
    model = GIN(input_dim=dataset.num_features, hidden_dim=16, output_dim=dataset.num_classes, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = "cpu"  # Change to "cuda" if a GPU is available
    model.to(device)

    # Train the model
    model.train()
    for epoch in range(30):  # Train for 30 epochs
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

    # Step 3: Evaluate the model
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            accuracy += (preds == data.y).sum().item()
    accuracy /= len(test_dataset)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # Step 4: Extract Concepts
    extractor = ConceptExtractor(beam_width=4, max_depth=2)
    data = next(iter(test_loader))  # Use a single batch for concept extraction
    data = data.to(device)
    with torch.no_grad():
        activations = model(data.x, data.edge_index, data.batch)
    labels = data.y
    base_concepts = extractor.extract_base_concepts(activations, labels)

    print("Extracted Base Concepts:", base_concepts)

    # Step 5: Visualize Graph Concept Activations
    graph_idx = 0  # Visualize the first graph in the batch
    # Visualize concept activations
    visualize_graph_concept_activations(
        data=data,
        activations=activations,
        concept=base_concepts[0],
        graph_idx=0,
        save_path="../results/concept_activation_map.png",
    )

    # Visualize chemical graph with names
    visualize_chemical_graph(
        data=data,
        activations=activations,
        concept=base_concepts[0],
        graph_idx=0,
        save_path="../results/chemical_graph.png",
    )

    print("Visualization saved to results/concept_activation_map.png")


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    main()
