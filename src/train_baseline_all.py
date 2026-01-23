# Baseline training runner for CIFAR-10, Fashion-MNIST, and CIFAR-100.

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.data_loaders import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_fashion_mnist_loaders,
)
from src.metrics import (
    MetricsLogger,
    compute_system_metrics,
    plot_comparison_curves,
    plot_learning_curves,
    reset_cuda_peak_memory,
)
from src.models import CNN3Layer, count_parameters
from src.trainer import save_checkpoint, train_epoch, validate_epoch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _save_metrics_csv(path: Path, epoch_metrics: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "grad_total_l2_norm"])
        for entry in epoch_metrics:
            gradients = entry.get("gradients", {}) or {}
            summary = gradients.get("summary", gradients)
            grad_total = summary.get("total_l2_norm") if isinstance(summary, dict) else None
            writer.writerow(
                [
                    entry.get("epoch"),
                    entry.get("train", {}).get("loss"),
                    entry.get("train", {}).get("accuracy"),
                    entry.get("validation", {}).get("loss"),
                    entry.get("validation", {}).get("accuracy"),
                    grad_total,
                ]
            )


def _collect_gradient_summary(grad_per_step: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not grad_per_step:
        return {}

    per_layer_accum: Dict[str, List[float]] = {}
    total_norms: List[float] = []
    zero_counts: List[int] = []

    for step_stats in grad_per_step:
        per_layer = step_stats.get("per_layer_l2_norms", {})
        for layer, value in per_layer.items():
            per_layer_accum.setdefault(layer, []).append(float(value))
        total_norms.append(float(step_stats.get("total_l2_norm", 0.0)))
        zero_counts.append(int(step_stats.get("zero_grad_parameters", 0)))

    per_layer_avg = {layer: float(sum(vals) / max(1, len(vals))) for layer, vals in per_layer_accum.items()}
    total_avg = float(sum(total_norms) / max(1, len(total_norms)))
    zero_avg = float(sum(zero_counts) / max(1, len(zero_counts)))

    return {
        "total_l2_norm": total_avg,
        "per_layer_l2_norms": per_layer_avg,
        "zero_grad_parameters": int(round(zero_avg)),
    }


def _dataset_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "cifar10",
            "dataset_label": "CIFAR-10",
            "loader_fn": get_cifar10_loaders,
            "num_classes": 10,
            "in_channels": 3,
        },
        {
            "name": "fashion_mnist",
            "dataset_label": "Fashion-MNIST",
            "loader_fn": get_fashion_mnist_loaders,
            "num_classes": 10,
            "in_channels": 1,
        },
        {
            "name": "cifar100",
            "dataset_label": "CIFAR-100",
            "loader_fn": get_cifar100_loaders,
            "num_classes": 100,
            "in_channels": 3,
        },
    ]


def _run_experiment(
    name: str,
    dataset_label: str,
    loader_fn,
    num_classes: int,
    in_channels: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    data_dir: str,
    use_amp: bool,
    use_compile: bool,
) -> Tuple[MetricsLogger, Path, Dict[str, Any]]:
    print(f"\n[Data] Loading {dataset_label}...")
    train_loader, val_loader = loader_fn(batch_size, num_workers, data_dir)

    inputs, targets = next(iter(train_loader))
    print(f"[Data] Batch shape: {tuple(inputs.shape)}, targets: {tuple(targets.shape)}")
    batch_mean = inputs.mean(dim=(0, 2, 3))
    batch_std = inputs.std(dim=(0, 2, 3))
    print(f"[Data] Batch mean: {batch_mean.tolist()}")
    print(f"[Data] Batch std:  {batch_std.tolist()}")

    model = CNN3Layer(num_classes=num_classes, in_channels=in_channels).to(device)
    total_params = count_parameters(model)
    print(f"[Model] {dataset_label} params: {total_params:,}")

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    run_metadata = {
        "dataset": dataset_label,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": 42,
        "amp": use_amp,
        "torch_compile": use_compile,
        "model": model.__class__.__name__,
    }
    logger = MetricsLogger(run_metadata=run_metadata)

    start_time = time.perf_counter()
    for epoch in tqdm(range(1, epochs + 1), desc=f"{dataset_label} epochs", unit="epoch"):
        print(f"\n[Epoch {epoch}/{epochs}] {dataset_label} starting...")
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
            collect_grad_stats_per_step=True,
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

        grad_summary = train_metrics.get("gradients")
        grad_per_step = train_metrics.get("gradients_per_step") or []
        grad_summary_from_steps = _collect_gradient_summary(grad_per_step)

        gradients_payload = {
            "summary": grad_summary or grad_summary_from_steps,
            "per_step": grad_per_step,
        }

        train_metrics_clean = {k: v for k, v in train_metrics.items() if k not in {"gradients", "gradients_per_step"}}

        logger.log_epoch(
            epoch=epoch,
            train=train_metrics_clean,
            validation=val_metrics,
            gradients=gradients_payload,
            system=system_metrics,
        )

        print(f"[Epoch {epoch}] Train: {_format_metrics(train_metrics_clean)}")
        print(f"[Epoch {epoch}] Val:   {_format_metrics(val_metrics)}")

        summary_norm = gradients_payload["summary"].get("total_l2_norm")
        if summary_norm is not None:
            if summary_norm > 1e3:
                print(f"[Warning] Potential gradient explosion: total L2 norm={summary_norm:.4e}")
            if summary_norm < 1e-6:
                print(f"[Warning] Potential gradient vanishing: total L2 norm={summary_norm:.4e}")

    elapsed = time.perf_counter() - start_time

    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{name}_metrics.json"
    logger.to_json(metrics_path)
    print(f"[Metrics] Saved {dataset_label} metrics to {metrics_path}")

    csv_path = metrics_dir / f"{name}_metrics.csv"
    _save_metrics_csv(csv_path, logger.epoch_metrics)
    print(f"[Metrics] Saved {dataset_label} CSV to {csv_path}")

    plot_learning_curves(metrics_path, output_dir=Path("results") / "figures", prefix=name)

    checkpoint_path = Path("checkpoints") / f"{name}_baseline.pth"
    final_metrics = {
        "train_loss": logger.epoch_metrics[-1]["train"].get("loss", 0.0),
        "train_accuracy": logger.epoch_metrics[-1]["train"].get("accuracy", 0.0),
        "val_loss": logger.epoch_metrics[-1]["validation"].get("loss", 0.0),
        "val_accuracy": logger.epoch_metrics[-1]["validation"].get("accuracy", 0.0),
    }
    save_checkpoint(str(checkpoint_path), model, optimizer, epochs, final_metrics)
    print(f"[Checkpoint] Saved {dataset_label} model to {checkpoint_path}")

    return logger, metrics_path, {
        "final_val_accuracy": final_metrics["val_accuracy"],
        "elapsed_sec": elapsed,
    }


def _find_convergence_epoch(val_acc: List[float]) -> int:
    if not val_acc:
        return 0
    best = max(val_acc)
    target = 0.9 * best
    for idx, value in enumerate(val_acc, start=1):
        if value >= target:
            return idx
    return len(val_acc)


def _gradient_stability(grad_norms: List[float]) -> float:
    if not grad_norms:
        return float("inf")
    mean = float(sum(grad_norms) / max(1, len(grad_norms)))
    if mean == 0:
        return float("inf")
    variance = sum((x - mean) ** 2 for x in grad_norms) / max(1, len(grad_norms))
    return (variance ** 0.5) / mean


def _write_report(
    report_path: Path,
    summary: Dict[str, Dict[str, Any]],
    comparison_plots: Dict[str, str],
) -> None:
    lines = [
        "# Dataset Comparison Report",
        "",
        "## Summary",
        "",
        "| Dataset | Final Val Acc | Total Time (s) | Convergence Epoch (90% of best) | Gradient Stability (CV) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for name, stats in summary.items():
        lines.append(
            f"| {stats['label']} | {stats['final_val_acc']:.4f} | {stats['elapsed_sec']:.1f} | {stats['convergence_epoch']} | {stats['grad_stability']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- **Fastest convergence:** " + summary[min(summary, key=lambda k: summary[k]["convergence_epoch"])]["label"],
            "- **Most stable gradients:** " + summary[min(summary, key=lambda k: summary[k]["grad_stability"])]["label"],
            "- **Dataset characteristics:**",
            "  - Fashion-MNIST is grayscale with 10 classes; simpler input modality tends to converge faster.",
            "  - CIFAR-10 and CIFAR-100 are RGB; CIFAR-100 has 100 classes and higher complexity, typically converging slower.",
            "  - Higher class count generally reduces final accuracy under identical hyperparameters.",
            "",
            "## Plots",
            "",
            f"- Accuracy comparison: {comparison_plots.get('accuracy_curve', '')}",
            f"- Loss comparison: {comparison_plots.get('loss_curve', '')}",
            f"- Gradient norm comparison: {comparison_plots.get('gradient_curves', '')}",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(42)

    lr = 1e-3
    batch_size = 128
    epochs = 5
    num_workers = 2
    data_dir = "assets"

    use_amp = device.type == "cuda"
    use_compile = hasattr(torch, "compile")

    print("[Setup] Device:", device)
    print("[Setup] AMP enabled:", use_amp)
    print("[Setup] torch.compile available:", use_compile)

    results: Dict[str, Dict[str, Any]] = {}
    metrics_paths: Dict[str, Path] = {}
    logs: Dict[str, MetricsLogger] = {}

    for cfg in _dataset_configs():
        logger, metrics_path, stats = _run_experiment(
            name=cfg["name"],
            dataset_label=cfg["dataset_label"],
            loader_fn=cfg["loader_fn"],
            num_classes=cfg["num_classes"],
            in_channels=cfg["in_channels"],
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            data_dir=data_dir,
            use_amp=use_amp,
            use_compile=use_compile,
        )
        logs[cfg["name"]] = logger
        metrics_paths[cfg["name"]] = metrics_path
        results[cfg["name"]] = stats

        target = 0.88 if cfg["name"] == "fashion_mnist" else (0.40 if cfg["name"] == "cifar100" else 0.65)
        if stats["final_val_accuracy"] < target:
            print(f"[Warning] {cfg['dataset_label']} target accuracy not met: {stats['final_val_accuracy']:.4f}")

    comparison_plots = plot_comparison_curves(
        {cfg["dataset_label"]: metrics_paths[cfg["name"]] for cfg in _dataset_configs()},
        output_dir="plots",
        prefix="baseline_part8",
    )
    print(f"[Plots] Saved comparison plots: {json.dumps(comparison_plots, indent=2)}")

    summary: Dict[str, Dict[str, Any]] = {}
    for cfg in _dataset_configs():
        metrics = logs[cfg["name"]].epoch_metrics
        val_acc = [entry.get("validation", {}).get("accuracy", 0.0) for entry in metrics]
        convergence_epoch = _find_convergence_epoch(val_acc)
        gradients = [
            (entry.get("gradients", {}) or {}).get("summary", {}).get("total_l2_norm")
            for entry in metrics
        ]
        grad_norms = [value for value in gradients if value is not None]
        grad_stability = _gradient_stability(grad_norms)

        summary[cfg["name"]] = {
            "label": cfg["dataset_label"],
            "final_val_acc": results[cfg["name"]]["final_val_accuracy"],
            "elapsed_sec": results[cfg["name"]]["elapsed_sec"],
            "convergence_epoch": convergence_epoch,
            "grad_stability": grad_stability,
        }

    report_path = Path("results") / "part8_report.md"
    _write_report(report_path, summary, comparison_plots)
    print(f"[Report] Saved to {report_path}")

    print("[Done]  Multi-dataset baselines complete.")


if __name__ == "__main__":
    main()