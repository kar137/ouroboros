# Data loading utilities for CIFAR-10, CIFAR-100, and Fashion-MNIST.

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Normalization statistics
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)
_FMNIST_MEAN = (0.2860,)
_FMNIST_STD = (0.3530,)


# Build transforms for CIFAR datasets.
def _build_cifar_transforms(train: bool) -> transforms.Compose:
    
    ops = []
    if train:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
        ]
    )
    return transforms.Compose(ops)

# Build transforms for Fashion-MNIST.
def _build_fmnist_transforms(train: bool) -> transforms.Compose:
    
    ops = []
    if train:
        ops.extend(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_FMNIST_MEAN, std=_FMNIST_STD),
        ]
    )
    return transforms.Compose(ops)


# Internal helper to create train/val loaders.
def _stratified_split_indices(
    targets: Iterable[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")

    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        per_class.setdefault(int(y), []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for _, indices in per_class.items():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _load_or_create_split_indices(
    dataset_name: str,
    targets: Iterable[int],
    val_fraction: float,
    split_seed: int,
    split_dir: Path,
) -> Tuple[List[int], List[int]]:

    split_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset_name}_seed{split_seed}_val{val_fraction:.3f}".replace(".", "p")
    split_path = split_dir / f"{tag}.json"

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["train_indices"], payload["val_indices"]

    train_indices, val_indices = _stratified_split_indices(targets, val_fraction, split_seed)
    payload = {
        "dataset": dataset_name,
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return train_indices, val_indices


def _make_loaders(
    dataset_cls,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
    val_fraction: float = 0.1,
    split_seed: Optional[int] = None,
    split_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    
    train_base = dataset_cls(root=data_dir, train=True, transform=train_transform, download=True)
    val_base = dataset_cls(root=data_dir, train=True, transform=val_transform, download=True)

    dataset_name = dataset_cls.__name__
    split_seed = int(split_seed if split_seed is not None else (seed if seed is not None else 42))
    split_dir_path = Path(split_dir) if split_dir is not None else Path(data_dir) / "splits"

    targets = getattr(train_base, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets; cannot create stratified split.")

    train_indices, val_indices = _load_or_create_split_indices(
        dataset_name=dataset_name,
        targets=targets,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_dir=split_dir_path,
    )

    train_set = torch.utils.data.Subset(train_base, train_indices)
    val_set = torch.utils.data.Subset(val_base, val_indices)

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def _seed_worker(worker_id: int) -> None:
            worker_seed = seed + worker_id
            torch.manual_seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        worker_init_fn = _seed_worker

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader

# Create CIFAR-10 training and validation DataLoaders.
def get_cifar10_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
    val_fraction: float = 0.1,
    split_seed: Optional[int] = None,
    split_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    
    train_tf = _build_cifar_transforms(train=True)
    val_tf = _build_cifar_transforms(train=False)
    return _make_loaders(
        datasets.CIFAR10,
        train_tf,
        val_tf,
        batch_size,
        num_workers,
        data_dir,
        seed=seed,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_dir=split_dir,
    )

# Create CIFAR-100 training and validation DataLoaders.
def get_cifar100_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
    val_fraction: float = 0.1,
    split_seed: Optional[int] = None,
    split_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
   
    train_tf = _build_cifar_transforms(train=True)
    val_tf = _build_cifar_transforms(train=False)
    return _make_loaders(
        datasets.CIFAR100,
        train_tf,
        val_tf,
        batch_size,
        num_workers,
        data_dir,
        seed=seed,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_dir=split_dir,
    )

# Create Fashion-MNIST training and validation DataLoaders.
def get_fashion_mnist_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
    val_fraction: float = 0.1,
    split_seed: Optional[int] = None,
    split_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    
    train_tf = _build_fmnist_transforms(train=True)
    val_tf = _build_fmnist_transforms(train=False)
    return _make_loaders(
        datasets.FashionMNIST,
        train_tf,
        val_tf,
        batch_size,
        num_workers,
        data_dir,
        seed=seed,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_dir=split_dir,
    )


# ============================================================================
# TEST SET LOADERS (for final evaluation)
# ============================================================================

def _make_test_loader(
    dataset_cls,
    transform,
    batch_size: int,
    num_workers: int,
    data_dir: str,
) -> DataLoader:
    """Create a test set DataLoader (no train split, just test=True)."""
    test_set = dataset_cls(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_cifar10_test_loader(batch_size: int, num_workers: int, data_dir: str) -> DataLoader:
    """Get CIFAR-10 test set loader."""
    return _make_test_loader(datasets.CIFAR10, _build_cifar_transforms(train=False), batch_size, num_workers, data_dir)


def get_cifar100_test_loader(batch_size: int, num_workers: int, data_dir: str) -> DataLoader:
    """Get CIFAR-100 test set loader."""
    return _make_test_loader(datasets.CIFAR100, _build_cifar_transforms(train=False), batch_size, num_workers, data_dir)


def get_fashion_mnist_test_loader(batch_size: int, num_workers: int, data_dir: str) -> DataLoader:
    """Get Fashion-MNIST test set loader."""
    return _make_test_loader(datasets.FashionMNIST, _build_fmnist_transforms(train=False), batch_size, num_workers, data_dir)
