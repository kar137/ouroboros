# Training and validation utilities with optional optimizations.

import time
from contextlib import nullcontext
from functools import partial
from weakref import WeakKeyDictionary
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader


# Compute top-1 accuracy for classification outputs.
def _compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return float(correct) / float(total) if total > 0 else 0.0


# Cache for compiled models to avoid recompilation overhead.
_compiled_model_cache = WeakKeyDictionary()


def maybe_compile_model(model: nn.Module, use_compile: bool) -> nn.Module:
    """Optionally wrap the model with torch.compile; safe fallback on failure."""

    if not use_compile:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model

    if model in _compiled_model_cache:
        return _compiled_model_cache[model]

    # Check if the model itself is already a compiled wrapper
    if hasattr(model, "_orig_mod"):
        return model

    try:
        compiled_model = compile_fn(model)
        _compiled_model_cache[model] = compiled_model
        return compiled_model
    except Exception:
        # Fall back silently to eager execution to preserve robustness.
        return model


# Create a warmup + cosine scheduler using SequentialLR.
def get_scheduler(optimizer: Optimizer, total_steps: int, warmup_steps: int) -> SequentialLR:

    warmup_steps = max(0, int(warmup_steps))
    cosine_steps = max(1, int(total_steps) - warmup_steps)

    if warmup_steps == 0:
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        return SequentialLR(optimizer, schedulers=[cosine], milestones=[])

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# Run one training epoch with optional AMP, accumulation, scheduler, compile.
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None,  # torch.amp.GradScaler or torch.cuda.amp.GradScaler
    scheduler: Optional[Any] = None,  # Any LR scheduler
    gradient_accumulation_steps: int = 1,
    amp_enabled: bool = False,
    use_compile: bool = False,
    collect_timing: bool = False,
    collect_grad_stats: bool = False,
    collect_grad_stats_per_step: bool = False,
    grad_stats_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    grad_stats_hook: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> Dict[str, float]:

    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

    compiled_model = maybe_compile_model(model, use_compile)
    if compiled_model is not model:
        model = compiled_model

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    grad_stats: Optional[Dict[str, float]] = None
    grad_stats_per_step: Optional[List[Dict[str, float]]] = [] if collect_grad_stats_per_step else None
    dataloader_len = len(dataloader)

    # Prepare AMP contexts.
    amp_active = amp_enabled and device.type == "cuda"

    if amp_active:
        # Prefer new torch.amp APIs (PyTorch 2.0+), but fall back for older versions.
        amp_autocast = getattr(getattr(torch, "amp", None), "autocast", None)
        if amp_autocast is not None:
            autocast_ctx = partial(amp_autocast, device_type="cuda")
        else:
            autocast_ctx = torch.cuda.amp.autocast

        amp_grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
        if scaler is not None:
            scaler_to_use = scaler
        elif amp_grad_scaler is not None:
            scaler_to_use = amp_grad_scaler("cuda")
        else:
            scaler_to_use = torch.cuda.amp.GradScaler()
    else:
        autocast_ctx = nullcontext
        scaler_to_use = scaler if (scaler is not None) else None

    start_time = time.perf_counter() if collect_timing else None

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    update_step = 0

    for step_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        is_last_batch = (step_idx + 1) == dataloader_len

        with autocast_ctx():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Preserve original loss magnitude for metrics before scaling/dividing.
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        loss = loss / float(gradient_accumulation_steps)

        if scaler_to_use is not None:
            scaler_to_use.scale(loss).backward()
        else:
            loss.backward()

        micro_step += 1
        is_update_step = (micro_step % gradient_accumulation_steps) == 0
        if is_update_step:
            if scaler_to_use is not None:
                scaler_to_use.step(optimizer)
                scaler_to_use.update()
            else:
                optimizer.step()

            update_step += 1

            if collect_grad_stats_per_step:
                if grad_stats_fn is None:
                    try:
                        from src.metrics import compute_gradient_stats  # type: ignore
                    except Exception:
                        from metrics import compute_gradient_stats  # type: ignore
                    grad_stats_fn = compute_gradient_stats
                step_stats = grad_stats_fn(model)
                if grad_stats_per_step is not None:
                    grad_stats_per_step.append(step_stats)
                if grad_stats_hook is not None:
                    grad_stats_hook(step_stats, update_step)

            if collect_grad_stats and is_last_batch:
                if grad_stats_fn is None:
                    try:
                        from src.metrics import compute_gradient_stats  # type: ignore
                    except Exception:
                        from metrics import compute_gradient_stats  # type: ignore
                    grad_stats_fn = compute_gradient_stats
                grad_stats = grad_stats_fn(model)

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

    # Flush remaining gradients if the final batch did not trigger an update.
    if micro_step > 0 and (micro_step % gradient_accumulation_steps) != 0:
        if scaler_to_use is not None:
            scaler_to_use.step(optimizer)
            scaler_to_use.update()
        else:
            optimizer.step()

        update_step += 1

        if collect_grad_stats_per_step:
            if grad_stats_fn is None:
                try:
                    from src.metrics import compute_gradient_stats  # type: ignore
                except Exception:
                    from metrics import compute_gradient_stats  # type: ignore
                grad_stats_fn = compute_gradient_stats
            step_stats = grad_stats_fn(model)
            if grad_stats_per_step is not None:
                grad_stats_per_step.append(step_stats)
            if grad_stats_hook is not None:
                grad_stats_hook(step_stats, update_step)

        if collect_grad_stats:
            if grad_stats_fn is None:
                try:
                    from src.metrics import compute_gradient_stats  # type: ignore
                except Exception:
                    from metrics import compute_gradient_stats  # type: ignore
                grad_stats_fn = compute_gradient_stats
            grad_stats = grad_stats_fn(model)

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

    avg_loss = running_loss / float(total_samples) if total_samples > 0 else 0.0
    avg_acc = running_correct / float(total_samples) if total_samples > 0 else 0.0

    metrics = {"loss": avg_loss, "accuracy": avg_acc}

    if grad_stats is not None:
        metrics["gradients"] = grad_stats

    if grad_stats_per_step is not None:
        metrics["gradients_per_step"] = grad_stats_per_step

    if collect_timing and start_time is not None:
        elapsed = time.perf_counter() - start_time
        metrics["time_sec"] = elapsed
        metrics["samples_per_sec"] = float(total_samples) / elapsed if elapsed > 0 else 0.0

    return metrics


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
    scheduler: Optional[Any] = None,  # Any LR scheduler
    scaler: Optional[Any] = None,  # torch.amp.GradScaler or torch.cuda.amp.GradScaler
) -> None:

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, path)


# Load checkpoint and restore model/optimizer state.
def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,  # Any LR scheduler
    scaler: Optional[Any] = None,  # torch.amp.GradScaler or torch.cuda.amp.GradScaler
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint