# Baseline CNN Architecture for Multi-Dataset Benchmarking: 3-layer convolutional neural network


import torch
import torch.nn as nn
from typing import Tuple


# 3-Layer Convolutional Neural Network with adaptive input support. 
class CNN3Layer(nn.Module):
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(CNN3Layer, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1)     # same padding: padding = kernel_size // 2  
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Spatial reduction: H/2, W/2
        
        # Block 2: Intermediate feature refinement
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # Spatial reduction: H/4, W/4
        
        # Block 3: High-level semantic features
        self.conv3 = nn.Conv2d(48, 88, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(88)
        self.relu3 = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))   # Resolution-agnostic pooling
        
        # Classifier head
        self.fc = nn.Linear(88, num_classes)
        
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
        x = torch.flatten(x, 1)  # Flatten from (N, 88, 1, 1) to (N, 88)
        x = self.fc(x)
        
        return x


# Count total trainable parameters in the model.
def count_parameters(model: nn.Module) -> int:  
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


