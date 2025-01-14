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
