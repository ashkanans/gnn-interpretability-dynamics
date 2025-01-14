import torch


class ActivationTracker:
    """
    Tracks activations of neurons in a GNN during forward passes.
    """

    def __init__(self):
        self.activations = {}

    def hook(self, module, input, output):
        """
        Hook function to capture activations.
        Args:
            module (torch.nn.Module): Layer being hooked.
            input (tuple): Input to the layer.
            output (torch.Tensor): Output of the layer.
        """
        self.activations[module] = output.detach().cpu()

    def register_hooks(self, model):
        """
        Registers hooks for all layers in the model.
        Args:
            model (torch.nn.Module): GNN model.
        """
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Module)):  # Add more layer types if needed
                module.register_forward_hook(self.hook)

    def clear(self):
        """
        Clears the stored activations.
        """
        self.activations.clear()

    def save_activations(self, file_path):
        """
        Saves activations to a file in a structured format.
        Args:
            file_path (str): Path to save the activations.
        """
        torch.save(self.activations, file_path)
        print(f"Activations saved to {file_path}")
