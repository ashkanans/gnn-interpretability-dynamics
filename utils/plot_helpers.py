import os

import torch


def plot_tradeoff(metrics, save_path, dataset_name):
    """
    Plot interpretability-accuracy trade-offs over training epochs.
    Args:
        metrics (dict): Metrics recorded during training.
        save_path (str): Path to save the plots.
        dataset_name (str): Name of the dataset.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epochs"], metrics["accuracy"], label="Accuracy", marker="o")
    plt.plot(metrics["epochs"], metrics["iou"], label="IoU (Interpretability)", marker="o")
    plt.plot(metrics["epochs"], metrics["abs"], label="ABS (Interpretability)", marker="o")
    plt.plot(metrics["epochs"], metrics["ent"], label="ENT (Interpretability)", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Scores")
    plt.title(f"Interpretability-Accuracy Trade-off on {dataset_name}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f"tradeoff_plot_{dataset_name}.png"))
    plt.show(block=True)


def visualize_comparison(results, save_path):
    """
    Visualize comparison results for interpretability metrics.
    Args:
        results (dict): Comparison results with metrics and baseline scores.
        save_path (str): Path to save the plot.
    """
    categories = ["Ease of Explanation", "Human Intuition", "IoU", "ABS", "ENT"]
    xgnn_scores = [results["xgnn_baseline"]["ease_of_explanation"], results["xgnn_baseline"]["intuition_alignment"]]
    our_scores = [
        results["our_metrics"]["iou"],
        results["our_metrics"]["abs"],
        results["our_metrics"]["ent"]
    ]

    # Bar chart comparison
    plt.figure(figsize=(10, 6))
    plt.bar(categories[:2], xgnn_scores, color="blue", label="XGNN Baseline", alpha=0.7)
    plt.bar(categories[2:], our_scores, color="green", label="Our Metrics", alpha=0.7)
    plt.ylabel("Score")
    plt.title("Comparison of Interpretability Metrics")
    plt.legend()
    plt.savefig(f"{save_path}/metric_comparison.png")
    plt.show(block=True)


def visualize_graph_concept_activations(data, activations, concept, graph_idx, save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx, subgraph

    # Step 1: Isolate the nodes and edges for the specific graph
    node_mask = (data.batch == graph_idx)  # Nodes belonging to graph_idx
    graph_node_indices = torch.where(node_mask)[0]  # Node indices for this graph
    graph_activations = activations[graph_node_indices]  # Node activations for this graph
    graph_concept = concept[graph_node_indices]  # Concept mask for this graph
    edge_index, edge_attr = subgraph(node_mask, data.edge_index, data.edge_attr, relabel_nodes=True)

    # Convert the subgraph to a NetworkX format
    subgraph_data = data.clone()
    subgraph_data.edge_index, subgraph_data.edge_attr = edge_index, edge_attr
    subgraph_data.x = data.x[graph_node_indices]
    subgraph_data.batch = data.batch[graph_node_indices]

    G = to_networkx(subgraph_data, to_undirected=True, remove_self_loops=True)

    # Normalize node activations for visualization
    node_activations = graph_activations.mean(dim=1).cpu().numpy()
    max_activation = node_activations.max()
    if max_activation > 0:
        node_activations /= max_activation

    # Convert concept mask to NumPy
    concept_activations = graph_concept.cpu().numpy()

    # Step 3: Assign node colors based on activations and concept alignment
    node_colors = []
    for i, activation in enumerate(node_activations):
        if concept_activations[i] > 0:
            # Highlight concept nodes in shades of red
            red_value = max(0.0, min(1.0, activation))  # Clamp to [0, 1]
            node_colors.append((1.0, 0.5 * red_value, 0.5 * red_value))
        else:
            # Non-concept nodes in shades of blue
            blue_value = max(0.0, min(1.0, activation))  # Clamp to [0, 1]
            node_colors.append((0.5 * blue_value, 0.5 * blue_value, 1.0))

    # Step 4: Assign edge colors based on node importance
    edge_colors = []
    for u, v in G.edges():
        u_index = list(G.nodes).index(u)
        v_index = list(G.nodes).index(v)
        edge_importance = (node_activations[u_index] + node_activations[v_index]) / 2
        edge_importance = max(0.0, min(1.0, edge_importance))  # Clamp to [0, 1]
        edge_colors.append((0.5 * edge_importance, 0.5, 0.5 * edge_importance))

    # Step 5: Plot the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Layout for node positions

    # Draw nodes with color mapping
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=300,
    )

    # Draw edges with color mapping
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=2,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add a title
    plt.title(f"Graph Concept Activation Map (Graph {graph_idx})", fontsize=16)

    # visualize_chemical_graph(data, activations, concept, graph_idx=0, save_path="chemical_graph.png")

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Graph visualization saved to {save_path}")
    else:
        plt.show()


import matplotlib.pyplot as plt


def visualize_chemical_graph(data, activations, concept, graph_idx, save_path=None):
    """
    Visualize a MUTAG graph with atom names, bond types, and concept activations.

    Args:
        data: DataBatch object containing the graph.
        activations: Tensor of neuron activations [num_nodes, num_neurons].
        concept: Boolean mask for concept [num_nodes].
        graph_idx: Index of the graph in the batch to visualize.
        save_path: Path to save the visualization (optional).
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx, subgraph

    # Step 1: Extract subgraph
    node_mask = (data.batch == graph_idx)  # Nodes belonging to graph_idx
    graph_node_indices = torch.where(node_mask)[0]  # Node indices for this graph
    graph_activations = activations[graph_node_indices]  # Node activations for this graph
    graph_concept = concept[graph_node_indices]  # Concept mask for this graph
    edge_index, edge_attr = subgraph(node_mask, data.edge_index, data.edge_attr, relabel_nodes=True)

    subgraph_data = data.clone()
    subgraph_data.edge_index, subgraph_data.edge_attr = edge_index, edge_attr
    subgraph_data.x = data.x[graph_node_indices]
    subgraph_data.batch = data.batch[graph_node_indices]

    G = to_networkx(subgraph_data, to_undirected=True, remove_self_loops=True)

    # Step 2: Define atom type colors and names
    atom_type_colors = {
        0: (0.8, 0.8, 0.8),  # Gray for Carbon (C)
        1: (0.5, 0.5, 1.0),  # Blue for Nitrogen (N)
        2: (1.0, 0.0, 0.0),  # Red for Oxygen (O)
        3: (0.0, 1.0, 0.0),  # Green for Fluorine (F)
        4: (0.5, 0.0, 0.5),  # Purple for Iodine (I)
        5: (1.0, 0.5, 0.0),  # Orange for Chlorine (Cl)
        6: (0.6, 0.3, 0.0),  # Brown for Bromine (Br)
    }
    atom_type_names = {
        0: "C",  # Carbon
        1: "N",  # Nitrogen
        2: "O",  # Oxygen
        3: "F",  # Fluorine
        4: "I",  # Iodine
        5: "Cl",  # Chlorine
        6: "Br",  # Bromine
    }

    # Edge types (bond types)
    bond_type_names = {
        0: "aromatic",
        1: "single",
        2: "double",
        3: "triple",
    }

    node_colors = []
    node_labels = {}
    for i, activation in enumerate(graph_activations.mean(dim=1).cpu().numpy()):
        atom_type = subgraph_data.x[i].argmax().item()
        node_colors.append(atom_type_colors.get(atom_type, (0.5, 0.5, 0.5)))  # Default to gray if unknown
        node_labels[i] = atom_type_names.get(atom_type, "?")  # Default to '?' if unknown

    # Step 3: Bond types (edge styles and labels)
    edge_styles = []
    edge_labels = {}
    for i, (u, v) in enumerate(G.edges()):
        bond_type = edge_attr[i].argmax().item() if i < edge_attr.size(0) else 1  # Default to single
        edge_styles.append("dashed" if bond_type == 0 else "solid")  # Dashed for aromatic
        edge_labels[(u, v)] = bond_type_names.get(bond_type, "single")

    # Step 4: Plot the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Layout for node positions

    # Draw nodes with atom type colors
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
    )

    # Draw edges with bond types
    for (u, v), style in zip(G.edges(), edge_styles):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            style=style,
            edge_color="black",
            width=2,
        )

    # Draw labels with atom names
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=10,
        font_color="black",
    )

    # Draw edge labels with bond types
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
    )

    # Add title
    plt.title(f"Chemical Graph (Graph {graph_idx})", fontsize=16)

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Chemical graph saved to {save_path}")
    else:
        plt.show()
