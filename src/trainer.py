# Training and validation utilities.

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Compute top-1 accuracy for classification outputs.
def _compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return float(correct) / float(total) if total > 0 else 0.0

# Run one training epoch.
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """ 
        Args:
        model: Model to train (already moved to the target device).
        dataloader: Training DataLoader yielding (inputs, targets).
        optimizer: Optimizer with parameters of ``model``.
        criterion: Loss function (e.g., ``nn.CrossEntropyLoss``).
        device: Torch device for computation.
    """

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / float(total_samples) if total_samples > 0 else 0.0
    avg_acc = running_correct / float(total_samples) if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": avg_acc}

# Run one validation epoch without gradient updates.
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    avg_loss = running_loss / float(total_samples) if total_samples > 0 else 0.0
    avg_acc = running_correct / float(total_samples) if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": avg_acc}

# Persist training state to disk.
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)


# Load checkpoint and restore model/optimizer state.
def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
