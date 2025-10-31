# BERT-based Vulnerability Detection

This implementation provides a BERT-based approach for detecting vulnerable smart contracts using the BCCC-VulSCs-2023 dataset.

## Overview

The BERT-based method fine-tunes a pre-trained BERT model on Solidity smart contract source code to classify them as:
- **Secure (0)**: No vulnerabilities detected
- **Vulnerable (1)**: Contract contains vulnerabilities

## Dataset

The BCCC-VulSCs-2023 dataset contains:
- **Total samples**: 36,670
- **Secure contracts**: 26,915
- **Vulnerable contracts**: 9,755
- **Features**: 70 columns including bytecode characteristics and AST features

## Implementation Details

### Architecture

The implementation uses `BertForSequenceClassification` from HuggingFace Transformers:
- **Base Model**: BERT-base-uncased (110M parameters)
- **Classification Head**: Binary classification (secure vs vulnerable)
- **Max Sequence Length**: 512 tokens
- **Fine-tuning**: All layers are fine-tuned on the smart contract data

### Key Components

1. **SmartContractDataset**: Custom PyTorch Dataset class for handling contract data
2. **BERTVulnerabilityDetector**: Main model class with training and evaluation methods
3. **Data Loading**: Supports both source code files and bytecode feature representations

### Training Configuration

Default hyperparameters:
```python
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
```

The model uses:
- **Optimizer**: AdamW with weight decay
- **Learning Rate Scheduler**: Linear warmup
- **Gradient Clipping**: Max norm of 1.0
- **Early Stopping**: Based on validation accuracy

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the training script:

```bash
python bert_vulnerability_detection.py
```

### With Source Code Files

If you have the actual Solidity source code files:

```python
from bert_vulnerability_detection import load_dataset, BERTVulnerabilityDetector

# Load data with source code
texts, labels = load_dataset(
    secure_path='BCCC-VolSCs-2023_Secure_.csv',
    vulnerable_path='BCCC-VolSCs-2023_Vulnerable.csv',
    source_code_dir='path/to/source/files'
)

# Initialize and train
detector = BERTVulnerabilityDetector()
train_loader, val_loader, test_loader = detector.prepare_data(texts, labels)
detector.train(train_loader, val_loader, epochs=5)
```

### Custom Configuration

```python
from bert_vulnerability_detection import BERTVulnerabilityDetector

# Initialize with custom parameters
detector = BERTVulnerabilityDetector(
    model_name='bert-base-uncased',
    num_labels=2
)

# Train with custom hyperparameters
detector.train(
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=3e-5,
    warmup_steps=500
)
```

### Making Predictions

```python
# Load trained model
detector.model.load_state_dict(torch.load('best_bert_model.pt'))

# Make predictions
predictions, true_labels, probabilities = detector.predict(test_loader)

# Get performance metrics
metrics = detector.get_metrics(true_labels, predictions, probabilities)
```

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for vulnerable class
- **Recall**: Recall for vulnerable class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

## Model Output

The trained model saves:
- `best_bert_model.pt`: Best model weights based on validation accuracy

## Performance Expectations

Expected performance ranges (actual results may vary):
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 75-90%
- **F1-Score**: 80-90%
- **ROC-AUC**: 85-95%

## Comparison with CodeBERT

| Aspect | BERT | CodeBERT |
|--------|------|----------|
| Pre-training | General text | Code-specific |
| Tokenization | WordPiece | BPE |
| Vocabulary | General English | Code + Comments |
| Best For | Text-like code | Programming code |

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM
- **Training Time**: 
  - CPU: ~2-4 hours (5 epochs)
  - GPU: ~20-40 minutes (5 epochs)

## File Structure

```
.
├── bert_vulnerability_detection.py  # Main implementation
├── requirements.txt                  # Python dependencies
├── BERT_IMPLEMENTATION.md           # This documentation
├── BCCC-VolSCs-2023_Secure_.csv    # Secure contracts dataset
├── BCCC-VolSCs-2023_Vulnerable.csv  # Vulnerable contracts dataset
└── best_bert_model.pt               # Saved model (after training)
```

## Extending the Implementation

### Adding Source Code Support

To use actual Solidity source code instead of bytecode features:

1. Organize your source files in a directory:
   ```
   source_code/
   ├── {hash_id_1}.sol
   ├── {hash_id_2}.sol
   └── ...
   ```

2. Modify the `load_dataset` call:
   ```python
   texts, labels = load_dataset(
       source_code_dir='source_code/'
   )
   ```

### Using Different BERT Variants

```python
# Use BERT Large
detector = BERTVulnerabilityDetector(model_name='bert-large-uncased')

# Use RoBERTa
detector = BERTVulnerabilityDetector(model_name='roberta-base')

# Use DistilBERT (faster, smaller)
detector = BERTVulnerabilityDetector(model_name='distilbert-base-uncased')
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce `BATCH_SIZE` (try 8 or 4)
2. Reduce `MAX_LENGTH` (try 256 or 128)
3. Use gradient accumulation:
   ```python
   accumulation_steps = 4
   loss = loss / accumulation_steps
   loss.backward()
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### Slow Training

To speed up training:
1. Use a GPU with CUDA
2. Reduce dataset size for experimentation
3. Use a smaller BERT variant (DistilBERT)
4. Increase batch size if memory allows

## References

- BERT Paper: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- Transformers Library: [HuggingFace](https://huggingface.co/transformers/)
- BCCC-VulSCs-2023 Dataset: Smart contract vulnerability detection dataset

## License

This implementation is provided as-is for research and educational purposes.
