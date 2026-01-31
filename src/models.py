# Baseline CNN Architecture for Multi-Dataset Benchmarking: 3-layer convolutional neural network


import torch
import torch.nn as nn
from typing import Tuple


# Default channel configuration for CIFAR-10 and Fashion-MNIST
DEFAULT_CHANNELS = [32, 64, 128]
# Wider channel configuration for CIFAR-100 (100 classes needs more capacity)
WIDE_CHANNELS = [64, 128, 256]


# 3-Layer Convolutional Neural Network with adaptive input support. 
class CNN3Layer(nn.Module):
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3, channels: Tuple[int, int, int] = None):
        super(CNN3Layer, self).__init__()
        
        # Use default channels if not specified
        if channels is None:
            channels = tuple(DEFAULT_CHANNELS)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        
        c1, c2, c3 = channels
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1)     # same padding: padding = kernel_size // 2  
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Spatial reduction: H/2, W/2
        
        # Block 2: Intermediate feature refinement
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # Spatial reduction: H/4, W/4
        
        # Block 3: High-level semantic features
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))   # Resolution-agnostic pooling
        
        # Classifier head
        self.fc = nn.Linear(c3, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass through the network.
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
               - CIFAR: (N, 3, 32, 32)
               - Fashion-MNIST: (N, 1, 28, 28)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.adaptive_pool(x)
        
        # Classifier
        x = torch.flatten(x, 1)  # Flatten from (N, C3, 1, 1) to (N, C3)
        x = self.fc(x)
        
        return x


# Count total trainable parameters in the model.
def count_parameters(model: nn.Module) -> int:  
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Print layer-wise output shapes and total parameter count.
def model_summary(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> None:
    """
    Args:
        model: PyTorch model to summarize
        input_shape: Input tensor shape (N, C, H, W)
    """
    print(f"\n{'='*70}")
    print(f"Model Summary - Input Shape: {input_shape}")
    print(f"{'='*70}\n")
    
    # Create dummy input
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    
    # Set model to eval mode for summary
    model.eval()
    
    # Hook to capture intermediate outputs
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.shape
        return hook
    
    # Register hooks for all layers
    hooks = []
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:  # Only leaf modules
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print layer information
    print(f"{'Layer Name':<30} {'Output Shape':<30}")
    print(f"{'-'*60}")
    for name, shape in activations.items():
        if name:     # Skip empty names
            print(f"{name:<30} {str(tuple(shape)):<30}")
    
    print(f"\n{'='*70}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"{'='*70}\n")


# Verification logic for multi-dataset compatibility.
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("CNN3Layer Architecture Verification")
    print("="*70)
    
    # Test 1: CIFAR-10 (RGB, 32×32) with default channels
    print("\n[Test 1] CIFAR-10 Configuration (default channels)")
    print("-" * 70)
    model_cifar = CNN3Layer(num_classes=10, in_channels=3)
    dummy_cifar = torch.randn(1, 3, 32, 32)
    
    output_cifar = model_cifar(dummy_cifar)
    param_count_cifar = count_parameters(model_cifar)
    
    print(f"Input Shape:      {tuple(dummy_cifar.shape)}")
    print(f"Output Shape:     {tuple(output_cifar.shape)}")
    print(f"Channels:         {model_cifar.channels}")
    print(f"Parameter Count:  {param_count_cifar:,}")
    print(f"Target Range:     80,000 - 120,000 parameters")
    print(f"Status:           {'✓ PASS' if 80000 <= param_count_cifar <= 120000 else '✗ FAIL'}")
    
    # Test 2: Fashion-MNIST (Grayscale, 28×28) with default channels
    print("\n[Test 2] Fashion-MNIST Configuration (default channels)")
    print("-" * 70)
    model_fmnist = CNN3Layer(num_classes=10, in_channels=1)
    dummy_fmnist = torch.randn(1, 1, 28, 28)
    
    output_fmnist = model_fmnist(dummy_fmnist)
    param_count_fmnist = count_parameters(model_fmnist)
    
    print(f"Input Shape:      {tuple(dummy_fmnist.shape)}")
    print(f"Output Shape:     {tuple(output_fmnist.shape)}")
    print(f"Channels:         {model_fmnist.channels}")
    print(f"Parameter Count:  {param_count_fmnist:,}")
    print(f"Target Range:     80,000 - 120,000 parameters")
    print(f"Status:           {'✓ PASS' if 80000 <= param_count_fmnist <= 120000 else '✗ FAIL'}")
    
    # Test 3: CIFAR-100 (RGB, 32×32) with WIDE channels
    print("\n[Test 3] CIFAR-100 Configuration (wide channels)")
    print("-" * 70)
    model_cifar100 = CNN3Layer(num_classes=100, in_channels=3, channels=WIDE_CHANNELS)
    dummy_cifar100 = torch.randn(1, 3, 32, 32)
    
    output_cifar100 = model_cifar100(dummy_cifar100)
    param_count_cifar100 = count_parameters(model_cifar100)
    
    print(f"Input Shape:      {tuple(dummy_cifar100.shape)}")
    print(f"Output Shape:     {tuple(output_cifar100.shape)}")
    print(f"Channels:         {model_cifar100.channels}")
    print(f"Parameter Count:  {param_count_cifar100:,}")
    print(f"Target Range:     300,000 - 500,000 parameters")
    print(f"Status:           {'✓ PASS' if 300000 <= param_count_cifar100 <= 500000 else '✗ FAIL'}")
    
    # Detailed architecture summary for CIFAR-10
    print("\n[Detailed Summary] CIFAR-10 Model")
    model_summary(model_cifar, (1, 3, 32, 32))
    
    # Detailed architecture summary for Fashion-MNIST
    print("\n[Detailed Summary] Fashion-MNIST Model")
    model_summary(model_fmnist, (1, 1, 28, 28))
    
    # Detailed architecture summary for CIFAR-100 (wide)
    print("\n[Detailed Summary] CIFAR-100 Model (wide)")
    model_summary(model_cifar100, (1, 3, 32, 32))
    
    print("\n" + "="*70)
    print("Verification Complete")
    print("="*70 + "\n")
