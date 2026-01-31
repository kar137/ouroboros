# Reproducibility suite for CIFAR-10, Fashion-MNIST, and CIFAR-100 baselines.

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
from torch.optim import AdamW
from tqdm import tqdm

from src.data_loaders import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_fashion_mnist_loaders,
)
from src.metrics import MetricsLogger, compute_system_metrics, plot_learning_curves, reset_cuda_peak_memory
from src.models import CNN3Layer, DEFAULT_CHANNELS, WIDE_CHANNELS, count_parameters
from src.trainer import save_checkpoint, train_epoch, validate_epoch, get_epoch_scheduler, get_current_lr


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Training hyperparameters
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1


def _dataset_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "cifar10",
            "dataset_label": "CIFAR-10",
            "loader_fn": get_cifar10_loaders,
            "num_classes": 10,
            "in_channels": 3,
            "target_acc": 0.65,
            "channels": DEFAULT_CHANNELS,  # [32, 64, 128]
        },
        {
            "name": "fashion_mnist",
            "dataset_label": "Fashion-MNIST",
            "loader_fn": get_fashion_mnist_loaders,
            "num_classes": 10,
            "in_channels": 1,
            "target_acc": 0.88,
            "channels": DEFAULT_CHANNELS,  # [32, 64, 128]
        },
        {
            "name": "cifar100",
            "dataset_label": "CIFAR-100",
            "loader_fn": get_cifar100_loaders,
            "num_classes": 100,
            "in_channels": 3,
            "target_acc": 0.40,
            "channels": WIDE_CHANNELS,  # [64, 128, 256] - wider for 100 classes
        },
    ]


def _run_single(
    dataset: Dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    data_dir: str,
    deterministic: bool,
) -> Dict[str, Any]:
    _set_seed(seed, deterministic)

    train_loader, val_loader = dataset["loader_fn"](batch_size, num_workers, data_dir, seed=seed)

    # Get channels config (use WIDE_CHANNELS for CIFAR-100, DEFAULT for others)
    channels = dataset.get("channels", DEFAULT_CHANNELS)
    
    model = CNN3Layer(
        num_classes=dataset["num_classes"],
        in_channels=dataset["in_channels"],
        channels=channels,
    ).to(device)
    
    # Use AdamW with weight decay instead of plain Adam
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Create epoch-level LR scheduler: warmup (1 epoch) + cosine annealing
    scheduler = get_epoch_scheduler(optimizer, total_epochs=epochs, warmup_epochs=WARMUP_EPOCHS)
    
    # Count parameters for logging
    total_params = count_parameters(model)

    run_metadata = {
        "dataset": dataset["dataset_label"],
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "deterministic": deterministic,
        "model": model.__class__.__name__,
        "channels": list(channels),
        "total_params": total_params,
    }
    logger = MetricsLogger(run_metadata=run_metadata)
    
    # Mandatory logging per user requirements
    print(f"[OPTIMIZER] Type: {type(optimizer).__name__} | "
          f"lr={optimizer.param_groups[0]['lr']} | "
          f"weight_decay={optimizer.param_groups[0]['weight_decay']}")
    print(f"[MODEL] Total params: {total_params}")
    print(f"[Model] {dataset['dataset_label']}: channels={channels}, params={total_params:,}")

    start_time = time.perf_counter()
    for epoch in tqdm(range(1, epochs + 1), desc=f"{dataset['dataset_label']} seed={seed}", unit="epoch"):
        reset_cuda_peak_memory()
        epoch_start = time.perf_counter()
        
        # Get current LR before training step
        current_lr = get_current_lr(optimizer)

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_enabled=(device.type == "cuda"),
            use_compile=hasattr(torch, "compile"),
            collect_grad_stats=True,
        )

        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Step the scheduler AFTER validation, BEFORE next epoch
        scheduler.step()
        
        # Log LR AFTER scheduler.step() to show the LR that will be used next epoch
        post_step_lr = optimizer.param_groups[0]['lr']
        print(f"[SCHEDULER] Epoch {epoch} | LR: {post_step_lr:.6f}")

        system_metrics = compute_system_metrics(
            total_samples=len(train_loader) * batch_size,
            start_time=epoch_start,
            end_time=time.perf_counter(),
            device=device,
        )

        train_metrics_clean = {k: v for k, v in train_metrics.items() if k != "gradients"}
        logger.log_epoch(
            epoch=epoch,
            train=train_metrics_clean,
            validation=val_metrics,
            gradients=train_metrics.get("gradients"),
            system=system_metrics,
            learning_rate=current_lr,
        )

    elapsed = time.perf_counter() - start_time

    results_dir = Path("results") / "reproducibility"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{dataset['name']}_seed{seed}_metrics.json"
    logger.to_json(metrics_path)

    checkpoint_path = Path("checkpoints") / f"{dataset['name']}_seed{seed}.pth"
    final_metrics = {
        "train_loss": logger.epoch_metrics[-1]["train"].get("loss", 0.0),
        "train_accuracy": logger.epoch_metrics[-1]["train"].get("accuracy", 0.0),
        "val_loss": logger.epoch_metrics[-1]["validation"].get("loss", 0.0),
        "val_accuracy": logger.epoch_metrics[-1]["validation"].get("accuracy", 0.0),
    }
    save_checkpoint(str(checkpoint_path), model, optimizer, epochs, final_metrics, scheduler=scheduler)

    plot_learning_curves(metrics_path, output_dir=Path("results") / "figures", prefix=f"{dataset['name']}_seed{seed}")

    return {
        "dataset": dataset["dataset_label"],
        "seed": seed,
        "final_training_accuracy": final_metrics["train_accuracy"],
        "final_validation_accuracy": final_metrics["val_accuracy"],
        "total_training_time": elapsed,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
    }


def run_reproducibility_suite(
    seeds: List[int],
    deterministic: bool = False,
    verify_determinism: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Dict[str, Any]] = []
    for dataset in _dataset_configs():
        for seed in seeds:
            results.append(
                _run_single(
                    dataset=dataset,
                    seed=seed,
                    device=device,
                    epochs=5,
                    batch_size=128,
                    lr=1e-3,
                    num_workers=2,
                    data_dir="assets",
                    deterministic=deterministic,
                )
            )

    deterministic_status = "not_run"
    if deterministic and verify_determinism:
        dataset = _dataset_configs()[0]
        seed = seeds[0]
        first = _run_single(
            dataset=dataset,
            seed=seed,
            device=device,
            epochs=5,
            batch_size=128,
            lr=1e-3,
            num_workers=2,
            data_dir="assets",
            deterministic=True,
        )
        second = _run_single(
            dataset=dataset,
            seed=seed,
            device=device,
            epochs=5,
            batch_size=128,
            lr=1e-3,
            num_workers=2,
            data_dir="assets",
            deterministic=True,
        )
        deterministic_status = (
            "passed"
            if first["final_validation_accuracy"] == second["final_validation_accuracy"]
            else "failed"
        )

    summary: List[Dict[str, Any]] = []
    for dataset in _dataset_configs():
        values = [
            r["final_validation_accuracy"]
            for r in results
            if r["dataset"] == dataset["dataset_label"]
        ]
        mean_acc = float(sum(values) / max(1, len(values)))
        variance = sum((x - mean_acc) ** 2 for x in values) / max(1, len(values))
        std_acc = float(variance ** 0.5)
        summary.append(
            {
                "dataset": dataset["dataset_label"],
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "min_accuracy": float(min(values)) if values else 0.0,
                "max_accuracy": float(max(values)) if values else 0.0,
                "target_accuracy": dataset["target_acc"],
            }
        )

    return results, summary, deterministic_status


def _write_csvs(run_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]) -> None:
    output_dir = Path("results") / "reproducibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _merge_fieldnames(preferred: List[str], rows: List[Dict[str, Any]]) -> List[str]:
        keys = []
        seen = set()
        for key in preferred:
            if key not in seen:
                keys.append(key)
                seen.add(key)
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys

    runs_path = Path("results") / "baseline_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_merge_fieldnames(
                [
                    "dataset",
                    "seed",
                    "final_training_accuracy",
                    "final_validation_accuracy",
                    "total_training_time",
                    "metrics_path",
                    "checkpoint_path",
                ],
                run_rows,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(run_rows)

    summary_path = Path("results") / "baseline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_merge_fieldnames(
                [
                    "dataset",
                    "mean_accuracy",
                    "std_accuracy",
                    "min_accuracy",
                    "max_accuracy",
                    "target_accuracy",
                ],
                summary_rows,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_report(
    summary_rows: List[Dict[str, Any]],
    run_rows: List[Dict[str, Any]],
    deterministic: bool,
    deterministic_status: str,
) -> None:
    max_var = max(summary_rows, key=lambda row: row["std_accuracy"])
    high_var_runs = [row for row in summary_rows if row["std_accuracy"] > 0.01]

    report_lines = [
        "# Reproducibility Report",
        "",
        f"Deterministic mode: **{deterministic}**",
        "",
        "## Summary",
        "",
        "| Dataset | Mean Acc | Std Acc | Min Acc | Max Acc |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in summary_rows:
        report_lines.append(
            f"| {row['dataset']} | {row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['min_accuracy']:.4f} | {row['max_accuracy']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Observations",
            "",
            f"- Highest variance: **{max_var['dataset']}** (std={max_var['std_accuracy']:.4f})",
            "- Seed-sensitive behavior: " + ("Yes" if high_var_runs else "No"),
            "- Deterministic check: " + deterministic_status,
        ]
    )

    report_path = Path("results") / "reproducibility" / "part9_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    seeds = [7, 42, 1337]
    deterministic = False
    verify_determinism = deterministic

    run_rows, summary_rows, deterministic_status = run_reproducibility_suite(
        seeds=seeds,
        deterministic=deterministic,
        verify_determinism=verify_determinism,
    )

    _write_csvs(run_rows, summary_rows)
    _write_report(summary_rows, run_rows, deterministic, deterministic_status)

    for row in summary_rows:
        if row["std_accuracy"] > 0.01:
            print(f"[Warning] {row['dataset']} std dev exceeds 1%: {row['std_accuracy']:.4f}")
        if row["mean_accuracy"] < row["target_accuracy"]:
            print(f"[Warning] {row['dataset']} mean accuracy below target: {row['mean_accuracy']:.4f}")

    print("[Done] Reproducibility suite complete.")


if __name__ == "__main__":
    main()