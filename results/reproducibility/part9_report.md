# Reproducibility Report

Deterministic mode: **False**

## Summary

| Dataset | Mean Acc | Std Acc | Min Acc | Max Acc |
| --- | --- | --- | --- | --- |
| CIFAR-10 | 0.6033 | 0.0049 | 0.5964 | 0.6074 |
| Fashion-MNIST | 0.8451 | 0.0045 | 0.8387 | 0.8483 |
| CIFAR-100 | 0.2668 | 0.0034 | 0.2623 | 0.2706 |

## Observations

- Highest variance: **CIFAR-10** (std=0.0049)
- Seed-sensitive behavior: No
- Deterministic check: not_run