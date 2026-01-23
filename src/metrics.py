# Lightweight metrics utilities for training/analysis.

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

# Compute gradient statistics for the given model.
def compute_gradient_stats(model: nn.Module) -> Dict[str, Any]:

    total_sq_norm = 0.0
    per_layer: Dict[str, float] = {}
    zero_grad_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue

            grad_data = grad.detach()

            if grad_data.is_sparse:
                values = grad_data.coalesce().values()
                layer_norm = float(torch.linalg.vector_norm(values).item()) if values.numel() > 0 else 0.0
            else:
                layer_norm = float(torch.linalg.vector_norm(grad_data).item())

            per_layer[name] = layer_norm
            total_sq_norm += layer_norm * layer_norm

            if math.isclose(layer_norm, 0.0, abs_tol=1e-12):
                zero_grad_params += 1

    total_l2_norm = math.sqrt(total_sq_norm) if total_sq_norm > 0.0 else 0.0

    return {
        "total_l2_norm": total_l2_norm,
        "per_layer_l2_norms": per_layer,
        "zero_grad_parameters": int(zero_grad_params),
    }


def reset_cuda_peak_memory() -> None:
    """Reset CUDA peak memory stats if available; no-op on CPU."""

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            # Some devices/backends may not support resetting stats; ignore.
            pass


def compute_system_metrics(
    total_samples: int,
    start_time: Optional[float],
    end_time: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Capture lightweight system metrics for an epoch."""

    metrics: Dict[str, Any] = {}

    if start_time is not None:
        end_t = time.perf_counter() if end_time is None else end_time
        elapsed = end_t - start_time
        metrics["epoch_time_sec"] = float(elapsed)
        metrics["throughput_samples_per_sec"] = float(total_samples) / elapsed if elapsed > 0 else 0.0

    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")

    metrics["device"] = str(device) if device is not None else "cpu"

    if torch.cuda.is_available():
        metrics["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
        metrics["max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2)

    return metrics


class MetricsLogger:
    """Accumulate structured metrics over epochs for later analysis."""

    def __init__(self, run_metadata: Optional[Dict[str, Any]] = None) -> None:
        self.run_metadata: Dict[str, Any] = run_metadata or {}
        self.epoch_metrics: List[Dict[str, Any]] = []
        self._prev_loss: Optional[float] = None
        self._prev_acc: Optional[float] = None

    def log_epoch(
        self,
        epoch: int,
        train: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
        gradients: Optional[Dict[str, Any]] = None,
        system: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a single epoch's metrics and compute convergence deltas."""

        val_metrics = validation or {}
        grad_metrics = gradients or {}
        system_metrics = system or {}

        loss_for_delta = val_metrics.get("loss", train.get("loss"))
        acc_for_delta = val_metrics.get("accuracy", train.get("accuracy"))

        loss_delta = None
        acc_improvement = None

        if loss_for_delta is not None and self._prev_loss is not None:
            loss_delta = float(loss_for_delta) - float(self._prev_loss)
        if acc_for_delta is not None and self._prev_acc is not None:
            acc_improvement = float(acc_for_delta) - float(self._prev_acc)

        if loss_for_delta is not None:
            self._prev_loss = float(loss_for_delta)
        if acc_for_delta is not None:
            self._prev_acc = float(acc_for_delta)

        entry: Dict[str, Any] = {
            "epoch": int(epoch),
            "train": {k: float(v) for k, v in train.items()},
            "validation": {k: float(v) for k, v in val_metrics.items()},
            "gradients": grad_metrics,
            "system": system_metrics,
            "convergence": {
                "loss_delta": loss_delta,
                "accuracy_improvement": acc_improvement,
            },
        }

        self.epoch_metrics.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics in the agreed JSON schema."""

        return {"run_metadata": self.run_metadata, "epoch_metrics": self.epoch_metrics}

    def to_json(self, path: Union[str, Path]) -> None:
        """Persist metrics to disk; intended to be called outside the train loop."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def reset(self) -> None:
        """Clear stored metrics while preserving run metadata."""

        self.epoch_metrics = []
        self._prev_loss = None
        self._prev_acc = None


def plot_learning_curves(
    metrics_json: Union[str, Path, Dict[str, Any]],
    output_dir: Union[str, Path] = "results/figures",
    prefix: str = "learning_curves",
) -> Dict[str, str]:
    """Generate loss/accuracy curves from stored metrics and save to disk."""

    if isinstance(metrics_json, (str, Path)):
        with Path(metrics_json).open("r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
    else:
        metrics_dict = metrics_json

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for entry in metrics_dict.get("epoch_metrics", []):
        epochs.append(entry.get("epoch"))
        train_loss.append(entry.get("train", {}).get("loss"))
        val_loss.append(entry.get("validation", {}).get("loss"))
        train_acc.append(entry.get("train", {}).get("accuracy"))
        val_acc.append(entry.get("validation", {}).get("accuracy"))

    if not epochs:
        raise ValueError("No epoch metrics available for plotting.")

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loss_path = output_path / f"{prefix}_loss.png"
    acc_path = output_path / f"{prefix}_accuracy.png"

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train loss", marker="o")
    if any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label="val loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss vs. Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, label="train acc", marker="o")
    if any(v is not None for v in val_acc):
        plt.plot(epochs, val_acc, label="val acc", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    return {"loss_curve": str(loss_path), "accuracy_curve": str(acc_path)}


def plot_comparison_curves(
    metrics_map: Dict[str, Union[str, Path, Dict[str, Any]]],
    output_dir: Union[str, Path] = "plots",
    prefix: str = "dataset_comparison",
) -> Dict[str, str]:
    """Plot comparison charts across datasets (loss, accuracy, gradient norms)."""

    def _load_metrics(value: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(value, (str, Path)):
            with Path(value).open("r", encoding="utf-8") as f:
                return json.load(f)
        return value

    metrics_by_dataset = {name: _load_metrics(data) for name, data in metrics_map.items()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Accuracy vs epoch
    acc_path = output_path / f"{prefix}_accuracy.png"
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for name, metrics in metrics_by_dataset.items():
        epochs = [entry.get("epoch") for entry in metrics.get("epoch_metrics", [])]
        accs = [entry.get("validation", {}).get("accuracy") for entry in metrics.get("epoch_metrics", [])]
        plt.plot(epochs, accs, marker="o", label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # Loss vs epoch
    loss_path = output_path / f"{prefix}_loss.png"
    plt.figure(figsize=(7, 4))
    for name, metrics in metrics_by_dataset.items():
        epochs = [entry.get("epoch") for entry in metrics.get("epoch_metrics", [])]
        losses = [entry.get("validation", {}).get("loss") for entry in metrics.get("epoch_metrics", [])]
        plt.plot(epochs, losses, marker="o", label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title("Loss vs. Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # Gradient norm patterns per layer (dataset overlays)
    grad_path = output_path / f"{prefix}_gradients.png"
    # Collect per-layer series per dataset.
    layer_names = set()
    per_dataset_layer_series: Dict[str, Dict[str, List[float]]] = {}

    for name, metrics in metrics_by_dataset.items():
        layer_series: Dict[str, List[float]] = {}
        for entry in metrics.get("epoch_metrics", []):
            gradients = entry.get("gradients", {}) or {}

            per_step = gradients.get("per_step") if isinstance(gradients, dict) else None
            if per_step:
                # Average per-layer norms across steps for the epoch.
                step_layer_values: Dict[str, List[float]] = {}
                for step_stats in per_step:
                    per_layer = step_stats.get("per_layer_l2_norms", {})
                    for layer, value in per_layer.items():
                        step_layer_values.setdefault(layer, []).append(float(value))
                for layer, values in step_layer_values.items():
                    layer_series.setdefault(layer, []).append(float(sum(values) / max(1, len(values))))
            else:
                per_layer = gradients.get("per_layer_l2_norms", {}) if isinstance(gradients, dict) else {}
                for layer, value in per_layer.items():
                    layer_series.setdefault(layer, []).append(float(value))

        per_dataset_layer_series[name] = layer_series
        layer_names.update(layer_series.keys())

    if layer_names:
        layer_names_sorted = sorted(layer_names)
        num_layers = len(layer_names_sorted)
        cols = 2
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)

        for idx, layer in enumerate(layer_names_sorted):
            ax = axes[idx // cols][idx % cols]
            for name, layer_series in per_dataset_layer_series.items():
                series = layer_series.get(layer, [])
                epochs = list(range(1, len(series) + 1))
                if series:
                    ax.plot(epochs, series, marker="o", label=name)
            ax.set_title(f"Gradient L2 Norm: {layer}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("avg grad norm")
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend()

        for idx in range(num_layers, rows * cols):
            fig.delaxes(axes[idx // cols][idx % cols])

        fig.tight_layout()
        fig.savefig(grad_path, dpi=200)
        plt.close(fig)

    return {
        "accuracy_curve": str(acc_path),
        "loss_curve": str(loss_path),
        "gradient_curves": str(grad_path) if layer_names else "",
    }
