# Vulnerable-Smart-Contracts-Detection-using-ML-DL

Project topic for InfoSec. The BCCC-VulSCs-2023 dataset is a substantial collection for Solidity Smart Contracts (SCs) analysis, comprising 36,670 samples, each enriched with 70 feature columns.

## Recent Improvements (fortify.ipynb)

The notebook has been significantly improved to address several critical issues:

### ✅ Fixed Issues

1. **SWA Update BN Crash** - Fixed NameError where `update_bn` referenced undefined `X_train_loader`. Now properly creates and uses `train_loader` from DataLoader.

2. **Consistent Batching** - All training now uses `DataLoader` with mini-batches instead of passing full tensors, enabling proper BatchNorm updates and better scalability.

3. **Loss/Logit Consistency** - All models now output logits (no sigmoid in forward methods) and use `BCEWithLogitsLoss` consistently. Sigmoid is applied only during evaluation.

4. **Single Train/Val Split** - Removed duplicate/unused test split. Now uses a single consistent train/validation split throughout.

5. **Reproducibility** - Random seeds are set at the start with `set_seed(42)`, including PyTorch deterministic flags.

6. **CodeBERT Embedding Caching** - Embeddings are now cached to `/kaggle/working/` to avoid expensive recomputation on subsequent runs.

### 🚀 New Features

- **Helper Functions**:
  - `set_seed(seed)` - Sets all random seeds for reproducibility
  - `make_loaders(X_train, y_train, X_val, y_val, batch_size)` - Creates PyTorch DataLoaders
  - `compute_codebert_embeddings(df, code_col, cache_dir, batch_size)` - Computes or loads cached embeddings

- **Improved Documentation** - Added markdown cells explaining the approach, fixes, and usage instructions

- **Better Evaluation** - Consistent metric computation with explicit sigmoid application during evaluation

### 💡 Usage

**First run** (no cache):
```bash
# Computes embeddings and saves to cache
# Takes ~5-10 minutes depending on dataset size
```

**Subsequent runs** (with cache):
```bash
# Loads embeddings from cache
# Saves ~5-10 minutes
```

**To force recomputation**:
```bash
# Delete the cache files:
rm /kaggle/working/codebert_embeddings.npy
rm /kaggle/working/ids.npy
```

### 📊 Notebook Structure

- 17 cells (streamlined from 27)
- All training uses DataLoader mini-batches
- Consistent use of BCEWithLogitsLoss
- Proper SWA implementation
- Complete evaluation suite (Accuracy, Precision, Recall, F1, ROC-AUC, Calibration)

### 🧪 Testing

The notebook is ready for testing on Kaggle GPU runtime. All critical sections have been verified:
- ✓ No undefined variable references
- ✓ Proper DataLoader integration
- ✓ Consistent loss functions
- ✓ Models output logits only
- ✓ Evaluation applies sigmoid correctly
