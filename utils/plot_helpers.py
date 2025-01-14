import os

from matplotlib import pyplot as plt


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
