# Ouroboros -> Phase 1A Multi-Dataset CNN Baseline: Comprehensive Findings Report

## Abstract

Phase 1A establishes a compact 3-layer CNN baseline across CIFAR-10, Fashion-MNIST, and CIFAR-100 using a consistent optimization stack (AdamW + warmup->cosine schedule, optional AMP, opportunistic torch.compile). The experiments demonstrate stable optimization over a 5-epoch budget: losses decrease monotonically, accuracies rise smoothly, and gradient norms remain finite with no parameters exhibiting exactly-zero gradients in recorded snapshots. Cross-dataset performance aligns with expected task difficulty: Fashion-MNIST converges rapidly to high accuracy (~86%), CIFAR-10 reaches a moderate baseline (~64%), and CIFAR-100 remains comparatively underfit (~35%) despite increased model capacity.

**Critical evaluation protocol correction:** Validation is now a stratified hold-out split drawn from the training set (train=True), while the official torchvision test split (train=False) is reserved for final reporting. The latest multi-dataset reproducibility run (results/phase1/phase1_all_datasets_results.json, 2026-02-01) reflects this fix: final validation and test accuracies are close but not numerically identical, consistent with distinct evaluation splits.

**Key finding:** Learning rate is highly influential in the 5-epoch regime. Increasing from lr=1e-3 to lr=3e-3 consistently improves accuracy across all datasets, indicating the baseline configuration is conservative for short-horizon optimization.

**Readiness for Phase 2:** The baseline is scientifically usable provided the project standardizes validation split construction (fixed split_seed per dataset) to eliminate remaining confounding in reproducibility comparisons.

---

## 1. Experimental Setup

### 1.1 Model Architecture and Parameter Scale

**Architecture:** 3 convolution blocks (Conv2d → BatchNorm → ReLU) with max-pooling after blocks 1-2, adaptive average pooling before classifier, and a linear classification head (src/models.py).

**Channel configurations and parameter counts:**
- **CIFAR-10:** [32, 64, 128] channels → 94,986 parameters
- **Fashion-MNIST:** [32, 64, 128] channels → 94,410 parameters  
- **CIFAR-100:** [64, 128, 256] channels → 397,412 parameters (4× wider to handle 100-class complexity)

**Design rationale:** Input-agnostic design via in_channels parameter and adaptive pooling supports both RGB 32×32 (CIFAR) and grayscale 28×28 (Fashion-MNIST) without resolution-specific flatten dimensions.

### 1.2 Dataset Characteristics and Preprocessing

**CIFAR-10 / CIFAR-100:**
- RGB 32×32 natural images
- Normalization: fixed per-channel mean/std
- Training augmentations: random crop (padding=4) + random horizontal flip (src/data_loaders.py)

**Fashion-MNIST:**
- Grayscale 28×28 clothing items
- Analogous pipeline: random crop + horizontal flip + normalization (src/data_loaders.py)
- **Note:** Horizontal flipping may be semantically ambiguous for asymmetric garments (e.g., left/right shoes), but empirically the model converges strongly in all runs.

### 1.3 Training Configuration

**Baseline hyperparameters** (from results/outputs/*.json and reproducibility suite):
- **Epochs:** 5
- **Batch size:** 128
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Warmup (start_factor=0.5, 1 epoch) → Cosine decay (eta_min=1e-5) via SequentialLR (src/trainer.py)
- **Hardware:** CUDA-enabled runs with AMP where available

**Reproducibility variants** (results/phase1/phase1_all_datasets_results.json):
- Per dataset: 5 experiments total
  - 3 seeds × baseline (AdamW lr=1e-3)
  - 1 run × higher LR (AdamW lr=3e-3)
  - 1 run × SGD (lr=1e-2, momentum=0.9)

### 1.4 Optimization Stack

**Automatic Mixed Precision (AMP):**
- Enabled on CUDA via torch.amp.autocast + GradScaler
- Fallback logic ensures graceful degradation if unavailable (src/trainer.py)

**torch.compile:**
- Enabled opportunistically; cached per model instance
- Silent fallback to eager execution on failure
- **Important caveat:** "Compile on/off" is not deterministic by configuration, actual behavior depends on runtime availability and success

**Determinism:**
- Seeds are set for reproducibility
- deterministic=False in reproducibility runner (deterministic CUDA settings not enforced)
- Observed low variance suggests "reliable enough" for baseline characterization, but not fully deterministic

### 1.5 Evaluation Protocol (Post-Fix)

**Current implementation** (src/data_loaders.py):

1. **Training pool:** torchvision train=True
2. **Validation:** Stratified subset of training pool (default val_fraction=0.1)
   - Indices persisted to data_dir/splits for reproducibility
   - Stratification ensures class balance
3. **Test:** torchvision train=False, accessed only via get_*_test_loader functions

**Critical reproducibility nuance:**
- split_seed defaults to the training seed if not explicitly set
- In the reproducibility notebook, each experiment passes seed=config["seed"]
- This means different seeds may induce different train/val splits **unless split_seed is fixed**
- **Implication:** Measured "seed variance" conflates training stochasticity with validation split variation

**Evidence of distinct val/test splits:**
- In results/phase1/phase1_all_datasets_results.json, final_val_acc and test_acc are close but not identical
  - Example: CIFAR-10 seed42 shows final_val_acc ≈ 0.6386 vs test_acc ≈ 0.6376
  - This is expected behavior under a correct protocol

**Historical context:**
- Earlier per-epoch logs (results/outputs/*.json) were generated before the split fix
- Those logs used test split as "validation" and should be interpreted as evaluation-on-test curves
- Optimization/gradient stability evidence from those logs remains valid

---

## 2. Results and Training Dynamics

### 2.1 Per-Dataset Performance Summary

**Baseline (AdamW lr=1e-3, 3-seed average):**

| Dataset | Test Acc (mean ± std) | Val Acc (mean ± std) | Interpretation |
|---------|----------------------|---------------------|----------------|
| CIFAR-10 | 0.641 ± 0.007 | 0.638 ± 0.005 | Moderate baseline; modest seed sensitivity |
| Fashion-MNIST | 0.863 ± 0.004 | 0.863 ± 0.004 | Strong convergence; low variance |
| CIFAR-100 | 0.349 ± 0.001 | 0.356 ± 0.006 | Underfit but very tight variance |

**Source:** results/phase1/phase1_all_datasets_results.json (2026-02-01)

### 2.2 Training Dynamics (From Per-Epoch Baseline Logs)

These single-run logs (results/outputs/*.json) provide the clearest view of optimization behavior over time.

**CIFAR-10:**
- **Loss trajectory:** Smooth decrease from ~1.60 (epoch 1) → ~0.99 (epoch 5)
- **Accuracy:** Rises from ~0.43 train / ~0.48 eval (epoch 1) → ~0.65 train / ~0.65 eval (epoch 5)
- **Curve shape:** Strong early learning (epochs 1-3), diminishing returns by epochs 4-5
- **Throughput:** ~1.9k-2.4k samples/sec after initial warmup
- **Memory:** Peak allocated ~60 MB, reserved ~116 MB

**Fashion-MNIST:**
- **Loss trajectory:** Rapid decrease from ~0.90 → ~0.37 over 5 epochs
- **Accuracy:** Rises from ~0.72 train / ~0.77 eval (epoch 1) → ~0.87 train / ~0.86 eval (epoch 5)
- **Convergence:** Gains slow notably by epoch 5; approaching plateau under current recipe
- **Timing anomaly:** First epoch unusually slow (~55s) vs later epochs (~23s)
  - Likely due to first-epoch overhead: compile graph capture, dataloader warmup, caching
  - This pattern also appears in CIFAR-100, supporting a systemic explanation
- **Memory:** Peak allocated ~57 MB, reserved ~116 MB

**CIFAR-100:**
- **Loss trajectory:** Decreases from ~3.80 → ~2.50
- **Accuracy:** Increases from ~0.13 train / ~0.17 eval (epoch 1) → ~0.36 train / ~0.36 eval (epoch 5)
- **Interpretation:** Clear underfitting within 5-epoch budget despite 4× parameter increase
  - Characteristic of insufficient optimization time for 100-class fine-grained task
  - Learning curves show continued improvement without saturation
- **Memory:** Peak allocated ~105 MB, reserved ~236 MB (consistent with wider channels)
- **Throughput:** ~1.1k-2.2k samples/sec

### 2.3 Cross-Dataset Comparison

**Convergence speed ranking:**
1. **Fashion-MNIST:** Rapid early gains, approaching saturation by epoch 5
2. **CIFAR-10:** Steady improvement, not yet saturated but gains diminishing
3. **CIFAR-100:** Continuous improvement but far from saturation

**Task difficulty vs. modality:**
- Grayscale 28×28 (Fashion-MNIST): Higher throughput, easier optimization
- RGB 32×32 natural images (CIFAR): Harder, especially CIFAR-100's fine-grained 100-class problem

**Consistency:** Per-epoch curves align with visual summaries in plots/ (accuracy/loss/gradients)

### 2.4 Learning Rate Sensitivity

**AdamW lr=3e-3 vs lr=1e-3 (single run per dataset):**

| Dataset | lr=1e-3 (baseline) | lr=3e-3 | Improvement |
|---------|-------------------|---------|-------------|
| CIFAR-10 | 0.641 | 0.685 | +6.9% relative |
| Fashion-MNIST | 0.863 | 0.882 | +2.2% relative |
| CIFAR-100 | 0.349 | 0.397 | +13.8% relative |

**Interpretation:**
- Consistent improvement across all datasets within the same 5-epoch budget
- Baseline lr=1e-3 is conservative for short-horizon optimization
- CIFAR-100 shows largest relative gains, suggesting it benefits most from accelerated early learning
- **Key insight:** Learning rate is a high-leverage knob for Phase 1b experiments

### 2.5 Optimizer Comparison: AdamW vs SGD

**SGD (lr=1e-2, momentum=0.9) performance (single run per dataset):**

| Dataset | AdamW lr=1e-3 | SGD lr=1e-2 | Delta |
|---------|---------------|-------------|-------|
| CIFAR-10 | 0.641 | 0.610 | -4.8% |
| Fashion-MNIST | 0.863 | 0.850 | -1.5% |
| CIFAR-100 | 0.349 | 0.260 | -25.5% |

**Interpretation:**
- SGD underperforms AdamW across all datasets in the 5-epoch regime
- May require longer schedules, different LR tuning, or augmentation adjustments
- **Conclusion:** AdamW is the strong default for short training budgets in this architecture family

### 2.6 Time-to-Convergence and Throughput

**From reproducibility summary** (results/phase1/phase1_all_datasets_results.json):
- **Mean epoch time:** ~17.8-19.2 seconds (stable across datasets/experiments)
- **Total 5-epoch time:** ~86-96 seconds
- Hardware-dependent but consistent within run set

**Throughput patterns** (from per-epoch logs):
- **First-epoch overhead:** Visible across all datasets (compile, caching, dataloader warmup)
- **Steady-state throughput:** Consistent from epoch 2 onward
- **Dataset ordering:** Fashion-MNIST (fastest) > CIFAR-10 > CIFAR-100 (slowest, due to wider model)

---

## 3. Optimization and Gradient Analysis

### 3.1 Gradient Norm Behavior

**Methodology note:** Gradient statistics are computed from the model's gradients at the **final update step of each epoch**, not averaged over all batches. This makes them sensitive to batch composition and explains some late-epoch spikes. These should be interpreted as **health checks** rather than comprehensive gradient-dynamics summaries.

**Key findings across datasets:**

1. **No vanishing gradients:** zero_grad_parameters = 0 in all logged snapshots
2. **No exploding gradients:** Losses decrease smoothly; norms remain finite
3. **Bounded and non-zero:** Total gradient L2 norms consistently positive and finite

**Dataset-specific patterns:**

**CIFAR-10:**
- Total gradient L2 norm: ~7.0 (epoch 1) → ~3.6 (epoch 3) → spike to ~8.4 (epoch 5)
- **Interpretation:** Spike at epoch 5 is consistent with end-of-epoch snapshot variability, not sustained instability
- Training remains stable throughout (monotonic loss decrease)

**Fashion-MNIST:**
- Total gradient L2 norm: Narrow band ~1.8-2.3, decreasing to ~1.76 by epoch 5
- **Interpretation:** Stable, well-conditioned optimization approaching plateau
- Consistent with rapid convergence observed in accuracy curves

**CIFAR-100:**
- Larger variability: ~2.7 early, notable spike ~10.3 at epoch 4
- Optimization remains monotonic despite spikes
- **Interpretation:** Snapshot noise + harder optimization landscape; spikes do not indicate pathology

### 3.2 Per-Layer Gradient Structure

**Common pattern** (from per_layer_l2_norms fields):
- **First convolution layer** (conv1.weight) typically dominates per-layer gradient norm
- **Later blocks and classifier:** Smaller but consistently non-zero contributions
- **Bias gradients:** Extremely small relative to weights (expected behavior)

**Interpretation:**
- Pattern is consistent with healthy signal propagation through shallow CNN
- Batch normalization likely contributes to stable gradients across layers
- No evidence of gradient flow pathologies

### 3.3 AMP and Compilation Effects

**What we can conclude:**
- AMP enabled on CUDA; no instability observed (no divergence, finite gradients)
- Pipeline is robust to compilation availability (graceful fallback)

**What we cannot conclude from current artifacts:**
- No paired ablations (AMP on/off, compile on/off) under identical seeds/splits
- Cannot quantify speedup or accuracy impact from AMP/compile alone
- "torch.compile enabled" reflects availability, not guaranteed usage (silent fallback)

**Recommendation for Phase 1b:** Include explicit AMP/compile ablations if these features will be systematically used in Phase 2.

---

## 4. Reliability and Reproducibility

### 4.1 Multi-Seed Consistency

**3-seed baseline standard deviations** (results/phase1/phase1_all_datasets_results.json):

| Dataset | Test Std | Val Std | Interpretation |
|---------|----------|---------|----------------|
| CIFAR-10 | 0.0068 | 0.0050 | Noticeable but tight for 5 epochs |
| Fashion-MNIST | 0.0040 | 0.0040 | Very tight; highly reproducible |
| CIFAR-100 | 0.0014 | 0.0063 | Extremely tight test; moderate val |

**Overall assessment:**
- Within the 5-epoch regime and this model family, run-to-run variability is low
- Strong positive signal for using this setup as a stable measurement harness
- Sufficient for comparative Phase 2 experiments

### 4.2 Evidence of Correct Evaluation Protocol

**Validation ≠ Test (post-fix confirmation):**
- final_val_acc and test_acc are close but not numerically identical
- Example: CIFAR-10 seed42 → val=0.6386, test=0.6376
- Consistent with distinct held-out sets (expected behavior)

**Provenance note:**
- Latest reproducibility run (2026-02-01) uses corrected protocol
- Earlier per-epoch logs (results/outputs/*.json) used test-as-validation
  - Optimization/gradient evidence remains valid
  - Performance comparisons should reference latest run

---

## 5. Key Insights and Implications

### 5.1 Baseline Health and Validity

**Evidence of correct implementation:**
✓ Monotonic loss decrease across all datasets  
✓ Smooth accuracy improvements without oscillation  
✓ Bounded, non-zero gradient norms  
✓ Stable per-epoch throughput and memory  
✓ Correct train/val/test separation (post-fix)  

**Conclusion:** The training system functions correctly and consistently. The baseline is scientifically usable for Phase 2.

### 5.2 Five Epochs as an "Optimization Probe"

**What 5 epochs reveals:**
- Dataset difficulty ranking (Fashion-MNIST < CIFAR-10 < CIFAR-100)
- Optimizer/hyperparameter sensitivity (LR > optimizer choice)
- Implementation stability and correctness

**What 5 epochs does NOT reveal:**
- Final attainable performance (all curves show continued improvement)
- Overfitting behavior (insufficient training for saturation)
- Long-horizon optimization dynamics

**Use case:** Short schedules are valuable for meta-optimization and hyperparameter ranking, but conclusions about final performance are limited.

### 5.3 Learning Rate Dominates Short-Horizon Performance

**Evidence:**
- lr=3e-3 consistently outperforms lr=1e-3 across all datasets (+2% to +14% relative)
- LR choice dominates optimizer type (AdamW vs SGD) in 5-epoch regime
- CIFAR-100 shows largest relative sensitivity (13.8% improvement)

**Implications for Phase 1b:**
- Prioritize LR sweeps early in hyperparameter search
- Treat LR as a primary optimization axis
- Consider dataset-specific LR tuning (CIFAR-100 may benefit from higher LR)

### 5.4 Dataset-Specific Design Considerations

**Fashion-MNIST:**
- Rapid convergence; may saturate quickly under stronger recipes
- Good candidate for ablation studies (fast iteration, stable signal)
- May need regularization/augmentation rather than longer training

**CIFAR-10:**
- Balanced learning dynamics; continued gains through epoch 5
- Representative of "moderate difficulty" vision tasks
- Good balance of signal quality and computational cost

**CIFAR-100:**
- Clear underfitting at 5 epochs despite 4× parameters
- **Phase 2 requirements:** Longer schedules, stronger augmentation, or architectural changes
- Most sensitive to LR; highest potential for improvement

### 5.5 Gradient Diagnostics: Health Checks, Not Optimization Proofs

**What gradient norms confirm:**
- Numerical stability (finite, non-zero)
- Absence of obvious pathologies (vanishing/exploding)
- Healthy signal propagation through architecture

**What gradient norms do NOT prove:**
- Optimality of optimizer/schedule choice
- Convergence to good local minima
- Adequacy of training length

**Complementary evidence:** LR sensitivity results already demonstrate material headroom for improvement, confirming that baseline recipe is not optimal.

---

## 6. Limitations and Caveats

### 6.1 Artifact Coverage Gaps

**Missing from current artifacts:**
- No per-epoch learning curves for latest multi-seed reproducibility run
  - Only final metrics/timing in summary JSON
  - Training dynamics discussion relies on single-run baseline logs
- No explicit AMP/compile ablations
  - Can only conclude "no obvious instability" under used settings
  - Cannot quantify performance impact

### 6.2 Confounding Factors

**Validation split and training seed coupling:**
- Default behavior couples these unless overridden
- Conflates pure "seed variance" interpretation
- Addressed by recommended split_seed standardization

**First-epoch timing anomaly:**
- Documented but not fully explained
- Consistent across datasets, suggesting systemic cause
- Does not affect scientific conclusions about convergence

### 6.3 Scope Boundaries

**What Phase 1 does NOT cover:**
- Long-horizon training dynamics (>5 epochs)
- Regularization/augmentation ablations
- Architectural variations beyond channel widths
- Advanced optimization techniques (SAM, Lion, etc.)
- Multi-GPU scaling behavior

**Intentional limitations:** Phase 1 is designed as a baseline establishment and pipeline validation, not comprehensive hyperparameter optimization.

---

## 7. Conclusions and Phase Transition Readiness

### 7.1 Is the Baseline Valid and Trustworthy?

**Yes, for Phase 1's intended purpose:**

1. **Training pipeline is stable:**
   - Monotonic loss decrease; no divergence
   - Smooth accuracy improvements
   - Consistent throughput and memory usage

2. **Gradients are healthy:**
   - Non-zero in all recorded diagnostics
   - Bounded; no sustained instability
   - Per-layer structure is plausible

3. **Evaluation protocol is scientifically defensible:**
   - Validation derived from train split (stratified)
   - Test held out for final reporting
   - Val/test metrics appropriately distinct

4. **Cross-dataset performance aligns with expectations:**
   - Difficulty ranking: Fashion-MNIST < CIFAR-10 < CIFAR-100
   - Performance scales with model capacity and dataset complexity

### 7.2 Metrics Most Predictive of Final Performance

**Within this artifact set:**

1. **Early-epoch validation accuracy improvements (epochs 1-3):**
   - Largest gains occur early
   - Diminishing returns by epoch 4-5
   - Strong signal for ranking hyperparameter configurations

2. **Learning rate:**
   - Dominates optimizer choice in 5-epoch regime
   - Consistent impact across datasets
   - Should be primary tuning axis in Phase 1b

### 7.4 Final Assessment

Phase 1a successfully establishes a **scientifically valid, computationally efficient baseline** suitable for comparative experiments in Phase 1b and beyond. The training pipeline is demonstrably stable, the evaluation protocol is correct, and the multi-dataset results provide clear signal about optimizer/hyperparameter sensitivity.

**The baseline is ready for Phase 1b/2** with the recommended standardizations (fixed split_seed, explicit LR search, extended schedules for proper convergence assessment).

---

## Appendix: Artifact Provenance

**Primary data sources:**

1. **results/phase1/phase1_all_datasets_results.json** (2026-02-01)
   - Latest multi-dataset reproducibility run (post-fix)
   - 5 experiments per dataset (3 baseline seeds + 2 variants)
   - Final metrics: train/val/test accuracy, loss, timing, parameters

2. **results/outputs/*.json** (pre-fix, single runs)
   - cifar10_baseline_metrics.json
   - fashion_mnist_baseline_metrics.json
   - cifar100_baseline_metrics.json
   - Per-epoch learning curves, gradient norms, system metrics
   - **Note:** Generated before split fix; interpret as evaluation-on-test

3. **Code references:**
   - src/data_loaders.py: Dataset loading, splitting, augmentation
   - src/models.py: Architecture definitions
   - src/trainer.py: Training loop, AMP, scheduling
   - notebooks/10_reproducibility_study.ipynb: Multi-seed experiments

4. **Visual summaries:**
   - plots/: Accuracy/loss/gradient comparison plots

**Reproducibility context:**
- Training seeds: 42, 123, 456 (baseline)
- Split seeds: Coupled to training seeds unless overridden
- Hardware: CUDA-enabled (specific device not logged)
- Software: PyTorch with torchvision datasets
