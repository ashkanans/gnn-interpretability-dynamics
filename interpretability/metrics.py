import torch


def compute_iou(activations, concept):
    """
    Compute Intersection over Union (IoU) between activations and concept.
    """
    threshold = torch.quantile(activations, 0.75, dim=0, keepdim=True)
    thresholded = (activations > threshold).float()

    intersection = (thresholded * concept.unsqueeze(1)).sum(dim=0)
    union = (thresholded + concept.unsqueeze(1) > 0).float().sum(dim=0)

    return (intersection / union).mean().item()


def compute_absolute_contribution(activations, concept):
    """
    Compute Absolute Contribution (ABS) of a concept to neuron activations.
    """
    concept_applied = activations * concept.unsqueeze(1)
    absolute_contribution = concept_applied.abs().sum(dim=0).mean().item()

    return absolute_contribution


def compute_entropy(concept):
    """
    Compute Entropy (ENT) of a concept mask.
    """
    probabilities = concept.float().mean().item()
    if probabilities == 0 or probabilities == 1:
        return 0.0

    entropy = -probabilities * torch.log2(torch.tensor(probabilities)) \
              - (1 - probabilities) * torch.log2(torch.tensor(1 - probabilities))
    return entropy.item()
