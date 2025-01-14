import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx


class GraphVisualizer:
    """
    Visualizes concept activation maps for graph datasets.
    """

    @staticmethod
    def node_activation_map(graph, activations, title="Node Activation Map"):
        """
        Visualize node importance using a color map.
        Args:
            graph (torch_geometric.data.Data): Input graph data.
            activations (torch.Tensor): Node activation values [num_nodes].
            title (str): Title of the plot.
        """
        # Convert PyTorch Geometric graph to NetworkX
        nx_graph = to_networkx(graph, to_undirected=True)

        # Normalize activations for visualization
        activations = activations.detach().cpu().numpy()
        node_colors = activations / activations.max()  # Normalize to [0, 1]

        # Plot graph with node colors
        pos = nx.spring_layout(nx_graph)
        plt.figure(figsize=(8, 6))
        nx.draw(
            nx_graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            with_labels=True,
            node_size=500,
            edge_color="gray"
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=activations.min(), vmax=activations.max()))
        sm.set_array([])
        plt.colorbar(sm)
        plt.title(title)
        plt.show()

    @staticmethod
    def edge_activation_map(graph, activations, edge_importance, title="Edge Activation Map"):
        """
        Visualize edge importance using a color map.
        Args:
            graph (torch_geometric.data.Data): Input graph data.
            activations (torch.Tensor): Node activation values [num_nodes].
            edge_importance (torch.Tensor): Edge importance values [num_edges].
            title (str): Title of the plot.
        """
        # Convert PyTorch Geometric graph to NetworkX
        nx_graph = to_networkx(graph, to_undirected=True)

        # Normalize activations for visualization
        activations = activations.detach().cpu().numpy()
        node_colors = activations / activations.max()

        # Normalize edge importance for visualization
        edge_importance = edge_importance.detach().cpu().numpy()
        edge_colors = edge_importance / edge_importance.max()

        # Prepare edge color mapping
        edges = list(nx_graph.edges)
        edge_color_map = {edges[i]: edge_colors[i] for i in range(len(edges))}

        # Plot graph with node and edge colors
        pos = nx.spring_layout(nx_graph)
        plt.figure(figsize=(8, 6))
        nx.draw(
            nx_graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            with_labels=True,
            node_size=500,
            edge_color="gray"
        )
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edge_color=[edge_color_map[edge] for edge in nx_graph.edges],
            edge_cmap=plt.cm.plasma,
            width=2
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=activations.min(), vmax=activations.max()))
        sm.set_array([])
        plt.colorbar(sm)
        plt.title(title)
        plt.show()

    @staticmethod
    def overlay_concept_map(graph, activations, edge_importance, concept, title="Overlayed Concept Map"):
        """
        Overlay concept activation maps on node and edge importance visualizations.
        Args:
            graph (torch_geometric.data.Data): Input graph data.
            activations (torch.Tensor): Node activation values [num_nodes].
            edge_importance (torch.Tensor): Edge importance values [num_edges].
            concept (torch.Tensor): Binary mask representing concept nodes [num_nodes].
            title (str): Title of the plot.
        """
        # Convert PyTorch Geometric graph to NetworkX
        nx_graph = to_networkx(graph, to_undirected=True)

        # Normalize activations and edge importance
        activations = activations.detach().cpu().numpy()
        node_colors = activations / activations.max()

        edge_importance = edge_importance.detach().cpu().numpy()
        edge_colors = edge_importance / edge_importance.max()

        # Highlight concept nodes
        concept_nodes = concept.detach().cpu().numpy()

        # Prepare edge color mapping
        edges = list(nx_graph.edges)
        edge_color_map = {edges[i]: edge_colors[i] for i in range(len(edges))}

        # Plot graph
        pos = nx.spring_layout(nx_graph)
        plt.figure(figsize=(10, 8))
        nx.draw(
            nx_graph,
            pos,
            node_color=["red" if concept_nodes[i] else "blue" for i in range(len(node_colors))],
            cmap=plt.cm.viridis,
            with_labels=True,
            node_size=500,
            edge_color="gray"
        )
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edge_color=[edge_color_map[edge] for edge in nx_graph.edges],
            edge_cmap=plt.cm.plasma,
            width=2
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=activations.min(), vmax=activations.max()))
        sm.set_array([])
        plt.colorbar(sm)
        plt.title(title)
        plt.show()
