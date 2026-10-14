"""Importable custom classification objective for the native training example."""

import torch


def equal_class_loss(scores, labels, config):
    """Mean weighted per-example CE; weights use training counts only."""
    weights = scores.new_tensor(config["weights"])
    per_example = torch.nn.functional.cross_entropy(
        scores, labels.long(), reduction="none"
    )
    return (per_example * weights[labels.long()]).mean()
