
# This script wires together data loading, model instantiation, training, metrics logging, and checkpointing for the first baseline run.

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.data_loaders import get_cifar10_loaders
from src.metrics import MetricsLogger, compute_system_metrics, plot_learning_curves, reset_cuda_peak_memory
from src.models import count_parameters
from src.trainer import save_checkpoint, train_epoch, validate_epoch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_model(num_classes: int, in_channels: int) -> nn.Module:

        from src.models import CNN3Layer

        return CNN3Layer(num_classes=num_classes, in_channels=in_channels)


def _format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main() -> None:

    # 1. Imports & Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(42)

    # 2. Hyperparameters (Baseline)
    lr = 1e-3
    batch_size = 128
    epochs = 5
    num_classes = 10
    num_workers = 2
    data_dir = "assets"

    use_amp = device.type == "cuda"
    use_compile = hasattr(torch, "compile")

    print("[Setup] Device:", device)
    print("[Setup] AMP enabled:", use_amp)
    print("[Setup] torch.compile available:", use_compile)

    # 3. Data Loading
    print("[Data] Loading CIFAR-10...")
    train_loader, val_loader = get_cifar10_loaders(batch_size, num_workers, data_dir)

    # Verify one batch shape and normalization stats.
    inputs, targets = next(iter(train_loader))
    print(f"[Data] Batch shape: {tuple(inputs.shape)}, targets: {tuple(targets.shape)}")
    batch_mean = inputs.mean(dim=(0, 2, 3))
    batch_std = inputs.std(dim=(0, 2, 3))
    print(f"[Data] Batch mean: {batch_mean.tolist()}")
    print(f"[Data] Batch std:  {batch_std.tolist()}")

    # 4. Model Instantiation
    model = _select_model(num_classes=num_classes, in_channels=3)
    model = model.to(device)
    total_params = count_parameters(model)
    print(f"[Model] Total trainable parameters: {total_params:,}")

    # 5. Optimizer & Loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 6. Training Loop
    run_metadata = {
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": 42,
        "amp": use_amp,
        "torch_compile": use_compile,
        "model": model.__class__.__name__,
    }
    logger = MetricsLogger(run_metadata=run_metadata)

    results_dir = Path("../results")
    checkpoints_dir = Path("../checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        print(f"\n[Epoch {epoch}/{epochs}] Starting...")
        reset_cuda_peak_memory()
        epoch_start = time.perf_counter()

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_enabled=use_amp,
            use_compile=use_compile,
            collect_grad_stats=True,
        )

        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_end = time.perf_counter()
        total_samples = len(train_loader) * batch_size
        system_metrics = compute_system_metrics(total_samples, epoch_start, end_time=epoch_end, device=device)

        grad_metrics = train_metrics.get("gradients")
        train_metrics_clean = {k: v for k, v in train_metrics.items() if k != "gradients"}

        logger.log_epoch(
            epoch=epoch,
            train=train_metrics_clean,
            validation=val_metrics,
            gradients=grad_metrics,
            system=system_metrics,
        )

        print(f"[Epoch {epoch}] Train: {_format_metrics(train_metrics_clean)}")
        print(f"[Epoch {epoch}] Val:   {_format_metrics(val_metrics)}")

        # Gradient warnings (explosion/vanishing) if stats present.
        if grad_metrics and "total_l2_norm" in grad_metrics:
            total_norm = grad_metrics["total_l2_norm"]
            if total_norm > 1e3:
                print(f"[Warning] Potential gradient explosion: total L2 norm={total_norm:.4e}")
            if total_norm < 1e-6:
                print(f"[Warning] Potential gradient vanishing: total L2 norm={total_norm:.4e}")

    
    # 7. Metrics & Logging
    metrics_path = results_dir / "cifar10_baseline_metrics.json"
    logger.to_json(metrics_path)
    print(f"[Metrics] Saved metrics to {metrics_path}")

    plot_paths = plot_learning_curves(metrics_path, output_dir=results_dir / "figures", prefix="cifar10_baseline")
    print(f"[Metrics] Saved plots: {json.dumps(plot_paths, indent=2)}")

    # 8. Checkpointing
    final_metrics = {
        "train_loss": logger.epoch_metrics[-1]["train"].get("loss", 0.0),
        "train_accuracy": logger.epoch_metrics[-1]["train"].get("accuracy", 0.0),
        "val_loss": logger.epoch_metrics[-1]["validation"].get("loss", 0.0),
        "val_accuracy": logger.epoch_metrics[-1]["validation"].get("accuracy", 0.0),
    }
    checkpoint_path = checkpoints_dir / "cifar10_baseline.pth"
    save_checkpoint(str(checkpoint_path), model, optimizer, epochs, final_metrics)
    print(f"[Checkpoint] Saved to {checkpoint_path}")

    # 9. Validation & Verification
    final_val_acc = final_metrics["val_accuracy"]
    print(f"[Result] Final validation accuracy: {final_val_acc:.4f}")
    if final_val_acc < 0.65:
        print("[Result] Warning: validation accuracy below 65% target.")

    print("[Done] Baseline experiment complete.")


if __name__ == "__main__":
    main()
