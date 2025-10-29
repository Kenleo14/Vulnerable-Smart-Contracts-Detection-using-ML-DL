# Vulnerable-Smart-Contracts-Detection-using-ML-DL

Project topic for InfoSec. The BCCC-VulSCs-2023 dataset is a substantial collection for Solidity Smart Contracts (SCs) analysis, comprising 36,670 samples, each enriched with 70 feature columns.

## Implementations

### BERT-based Detection (NEW)

A BERT-based method for detecting vulnerable smart contracts. This implementation fine-tunes a pre-trained BERT model on raw source code of Solidity smart contracts to classify them as secure (0) or vulnerable (1).

**Features:**
- Fine-tuned BERT model for sequence classification
- Support for both source code and bytecode feature representations
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Easy-to-use API for training and inference

**Quick Start:**
```bash
pip install -r requirements.txt
python bert_vulnerability_detection.py
```

For detailed documentation, see [BERT_IMPLEMENTATION.md](BERT_IMPLEMENTATION.md)

### CodeBERT-based Detection

The original implementation using CodeBERT embeddings is available in `fortify.ipynb`.

## Dataset

- **Total samples**: 36,670
- **Secure contracts**: 26,915 (label: 0)
- **Vulnerable contracts**: 9,755 (label: 1)
- **Features**: 70 columns including bytecode characteristics and AST features

## Files

- `bert_vulnerability_detection.py` - BERT-based implementation
- `BERT_IMPLEMENTATION.md` - Detailed documentation for BERT method
- `requirements.txt` - Python dependencies
- `fortify.ipynb` - Original CodeBERT notebook implementation
- `BCCC-VolSCs-2023_Secure_.csv` - Secure contracts dataset
- `BCCC-VolSCs-2023_Vulnerable.csv` - Vulnerable contracts dataset
