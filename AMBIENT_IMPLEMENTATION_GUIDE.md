# Ambient Diffusion Implementation Guide

## 🎯 Overview

This implementation adds **Ambient Diffusion style training** to your time series project as an alternative to the TST completion approach. You can now choose between two training modes:

### **Mode 1: Original (TST + Masking)**
- Uses TST encoder/decoder to complete missing values
- Creates "natural neighborhoods" for convolution
- Single mask from NaN detection
- Network input: `C` channels

### **Mode 2: Ambient Diffusion (Dual Corruption)**
- **NO** TST completion
- Dual corruption matrices (A, Ã) like Ambient Diffusion paper
- Network input: `2*C` channels (image + mask concatenated)
- Three losses: train_loss (Ã), val_loss (A), test_loss (all)

---

## 📋 What Was Implemented

### 1. **New Arguments** (`utils/utils_args.py`)
```python
--use_ambient_style      # Enable/disable Ambient mode (default: False)
--delta_probability      # Further corruption δ (default: 0.1)
```

### 2. **Helper Function** (`utils/utils.py`)
```python
create_further_corruption(corruption_matrix, delta_probability, device)
```
Creates Ã = A ⊙ B where B is additional random corruption.

### 3. **Model Methods** (`models/our.py`)
```python
# New methods:
loss_fn_ambient(x, corruption_matrix, hat_corruption_matrix, sigma)
forward_ambient(x, hat_corruption_matrix, sigma, labels, augment_pipe)

# Modified:
__init__()  # Now supports 2*C input channels when use_ambient_style=True
```

### 4. **Training Loop** (`run_irregular.py`)
- Conditional TST initialization (only if not Ambient mode)
- Conditional TST pre-training (skipped in Ambient mode)
- Dual-mode training loop with clear separation
- Updated checkpoint saving (handles missing TST components)

---

## 🚀 How to Use

### **Option 1: Command Line**

#### Original Mode (TST + Masking):
```bash
python run_irregular.py \
    --config ./configs/seq_len_24/stock.yaml \
    --use_ambient_style False \
    --missing_rate 0.3
```

#### Ambient Mode (No TST, Dual Corruption):
```bash
python run_irregular.py \
    --config ./configs/seq_len_24/stock_ambient.yaml \
    --use_ambient_style True \
    --delta_probability 0.1 \
    --missing_rate 0.3
```

### **Option 2: Config File**

Use the provided `configs/seq_len_24/stock_ambient.yaml`:
```bash
python run_irregular.py --config ./configs/seq_len_24/stock_ambient.yaml
```

---

## 🔬 Technical Details

### **Data Flow Comparison**

#### Original Mode:
```
Irregular Data (NaN) 
  → Propagate values (forward/backward fill)
  → TST Encoder → TST Decoder
  → Completed time series
  → ts_to_img() → completed image
  → Single mask from original NaN
  → loss_fn_irregular(completed_img, mask)
```

#### Ambient Mode:
```
Irregular Data (NaN)
  → ts_to_img() → image with NaN
  → Create corruption_matrix (A): 1=observed, 0=NaN
  → Create hat_corruption_matrix (Ã): further corrupt A by δ
  → Replace NaN with 0
  → loss_fn_ambient(x, A, Ã, sigma=0.0)
  → Three losses computed, val_loss used for backprop
```

### **Network Input**

#### Original Mode:
- Input shape: `[B, C, H, W]` where C = input_channels (e.g., 6)
- Receives: Completed image

#### Ambient Mode:
- Input shape: `[B, 2*C, H, W]` where C = input_channels
- Receives: `[corrupted_image, hat_corruption_mask]` concatenated
- Channels 0 to C-1: Ã(x + ση) (corrupted noisy image)
- Channels C to 2C-1: Ã itself (the mask)

### **Loss Computation**

#### Original Mode:
```python
loss = mean(weight * mask * (output - x)²)
# Single loss on observed pixels only
```

#### Ambient Mode:
```python
train_loss = mean(weight * Ã * (output - x)²)      # On further corrupted
val_loss = mean(weight * A * (output - x)²)        # On original corruption ← USED
test_loss = mean(weight * (output - x)²)           # On all pixels
```

---

## 🎛️ Hyperparameters

### **Critical Parameters**

| Parameter | Original Mode | Ambient Mode | Notes |
|-----------|--------------|--------------|-------|
| `use_ambient_style` | `False` | `True` | Enables Ambient mode |
| `delta_probability` | N/A | `0.1` | Further corruption δ |
| `first_epoch` | `30` | `0` | TST pre-training (not needed in Ambient) |
| Network input | `C` channels | `2*C` channels | Automatic based on mode |

### **Recommended Settings for Ambient Mode**

```yaml
use_ambient_style: true
delta_probability: 0.1          # Start with 0.1, can try [0.05, 0.15, 0.2]
first_epoch: 0                  # No TST pre-training
learning_rate: 0.0001           # Same as original
batch_size: 128                 # Same as original
```

---

## 📊 Expected Behavior

### **During Training**

#### Ambient Mode Startup:
```
🚀 AMBIENT DIFFUSION MODE ENABLED
================================================================================
✓ TST encoder/decoder: DISABLED
✓ Dual corruption (A, Ã): ENABLED
✓ Delta probability (δ): 0.1
✓ Network input: 2 × 6 = 12 channels
================================================================================

🔧 Ambient Diffusion mode: Network expects 12 input channels
   (6 for image + 6 for mask)

⏭️  SKIPPING TST PRE-TRAINING (Ambient mode)
```

#### Loss Logging:
```
Epoch 10:
  train/train_loss: 0.0234    # Loss on Ã (monitoring)
  train/val_loss: 0.0189      # Loss on A (used for backprop)
  train/test_loss: 0.0512     # Loss on all pixels (evaluation)
```

### **Key Differences to Expect**

1. **Training Speed**: Ambient mode should be faster (no TST forward/backward passes)
2. **Memory Usage**: Similar or slightly higher (2*C input channels vs C + TST)
3. **Sample Quality**: May differ - Ambient relies on dual corruption, not completion

---

## 🧪 Testing Your Implementation

### **Quick Test (Single Batch)**
```python
# Add to run_irregular.py temporarily for debugging
if epoch == 0 and i == 1:
    print(f"x_img shape: {x_img.shape}")
    print(f"corruption_matrix mean: {corruption_matrix.mean():.3f}")
    print(f"hat_corruption_matrix mean: {hat_corruption_matrix.mean():.3f}")
    print(f"Expected ratio: {(1-args.missing_rate) * (1-args.delta_probability):.3f}")
```

### **Full Training Test**
```bash
# Train for 10 epochs to verify everything works
python run_irregular.py \
    --config ./configs/seq_len_24/stock_ambient.yaml \
    --epochs 10 \
    --logging_iter 5
```

---

## 📈 Comparison Experiments

### **Recommended Ablation Studies**

1. **Baseline vs Ambient**
   ```bash
   # Original mode
   python run_irregular.py --config configs/seq_len_24/stock.yaml \
       --use_ambient_style False --epochs 100
   
   # Ambient mode
   python run_irregular.py --config configs/seq_len_24/stock_ambient.yaml \
       --use_ambient_style True --epochs 100
   ```

2. **Delta Probability Sweep**
   ```bash
   for delta in 0.05 0.1 0.15 0.2; do
       python run_irregular.py --config configs/seq_len_24/stock_ambient.yaml \
           --delta_probability $delta --epochs 100
   done
   ```

3. **Missing Rate Comparison**
   ```bash
   for missing in 0.3 0.5 0.7; do
       python run_irregular.py --config configs/seq_len_24/stock_ambient.yaml \
           --missing_rate $missing --epochs 100
   done
   ```

---

## 🔍 Debugging Tips

### **Problem: Shape mismatch in network**
**Symptom**: `Expected input with X channels but got Y`

**Solution**: Check `use_ambient_style` flag is correctly set in config/args

### **Problem: NaN in loss**
**Symptom**: Loss becomes NaN during training

**Possible causes**:
1. All pixels masked (both A and Ã are all zeros)
2. Learning rate too high
3. Gradient explosion

**Solutions**:
```python
# Add to loss_fn_ambient for debugging
if torch.isnan(val_loss):
    print(f"NaN detected!")
    print(f"A visible: {corruption_matrix.sum()}/{corruption_matrix.numel()}")
    print(f"Ã visible: {hat_corruption_matrix.sum()}/{hat_corruption_matrix.numel()}")
```

### **Problem: val_loss >> train_loss**
**Symptom**: Large gap between val_loss and train_loss

**Expected**: This is NORMAL in Ambient mode!
- train_loss: computed on Ã (fewer pixels)
- val_loss: computed on A (more pixels)
- val_loss should be higher

---

## 🎓 Key Insights

### **Why Dual Corruption Works**

The key insight from Ambient Diffusion is:
> The model doesn't know if a pixel is missing because:
> 1. It was never observed (in dataset), OR
> 2. It was deliberately removed during training (further corruption)

This forces the model to learn ALL pixels, not just observed ones.

### **Comparison to Original Approach**

| Aspect | Original (TST) | Ambient |
|--------|---------------|---------|
| **Completion** | TST fills NaN → natural neighborhoods | No completion, NaN→0 |
| **Training signal** | Masked loss on completed image | Masked loss on A, input with Ã |
| **Philosophy** | "Complete, then don't trust completion" | "Corrupt further, force full learning" |
| **Computational cost** | TST forward/backward + diffusion | Diffusion only |

---

## 📝 Checklist for Your Run

Before training in Ambient mode, verify:

- [ ] `use_ambient_style: true` in config
- [ ] `delta_probability` is set (e.g., 0.1)
- [ ] `first_epoch: 0` (no TST pre-training needed)
- [ ] Correct config file used (e.g., `stock_ambient.yaml`)
- [ ] Checkpoint directory will have `_ambient` suffix
- [ ] Network expects `2 * input_channels` (printed at startup)
- [ ] Three losses logged: train_loss, val_loss, test_loss

---

## 🚨 Important Notes

1. **Inference Not Modified**: Sampling/generation code remains unchanged. You'll need to update it separately if you want to use trained Ambient models.

2. **Checkpoints Separate**: Ambient mode saves to `missing_rate_XX_ambient/` directory to avoid conflicts with original mode checkpoints.

3. **Mask Convention**: 
   - `1 = observed/visible`
   - `0 = missing/corrupted`
   (Same convention as Ambient Diffusion paper)

4. **Phase 1**: Current implementation uses `sigma=0.0` (no noise). This is intentional for initial testing. Can be changed to `sigma=None` for full noise schedule.

---

## 📚 References

- **Ambient Diffusion Paper**: "Learning Clean Distributions from Corrupted Data"
- **Implementation Files**:
  - `utils/utils_args.py`: Arguments
  - `utils/utils.py`: Helper functions
  - `models/our.py`: Model and loss functions
  - `run_irregular.py`: Training loop
  - `configs/seq_len_24/stock_ambient.yaml`: Example config

---

## ✅ Summary

You now have a complete implementation of Ambient Diffusion style training that:
- ✅ Works alongside original TST-based approach
- ✅ Supports dual corruption (A, Ã)
- ✅ Properly concatenates image and mask for network input
- ✅ Computes three losses (train, val, test)
- ✅ Uses val_loss for backpropagation
- ✅ Skips TST initialization and pre-training
- ✅ Saves checkpoints separately

**Next step**: Run a training experiment and compare results! 🚀

