# Data loading utilities for CIFAR-10, CIFAR-100, and Fashion-MNIST.

import random
from typing import Optional, Tuple

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
def _make_loaders(
    dataset_cls,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    
    train_set = dataset_cls(root=data_dir, train=True, transform=train_transform, download=True)
    val_set = dataset_cls(root=data_dir, train=False, transform=val_transform, download=True)

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
) -> Tuple[DataLoader, DataLoader]:
    
    train_tf = _build_cifar_transforms(train=True)
    val_tf = _build_cifar_transforms(train=False)
    return _make_loaders(datasets.CIFAR10, train_tf, val_tf, batch_size, num_workers, data_dir, seed=seed)

# Create CIFAR-100 training and validation DataLoaders.
def get_cifar100_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
   
    train_tf = _build_cifar_transforms(train=True)
    val_tf = _build_cifar_transforms(train=False)
    return _make_loaders(datasets.CIFAR100, train_tf, val_tf, batch_size, num_workers, data_dir, seed=seed)

# Create Fashion-MNIST training and validation DataLoaders.
def get_fashion_mnist_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    
    train_tf = _build_fmnist_transforms(train=True)
    val_tf = _build_fmnist_transforms(train=False)
    return _make_loaders(datasets.FashionMNIST, train_tf, val_tf, batch_size, num_workers, data_dir, seed=seed)
