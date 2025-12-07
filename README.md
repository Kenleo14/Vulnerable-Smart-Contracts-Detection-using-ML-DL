# Vulnerable Smart Contracts Detection using ML/DL

## Project Overview

This project focuses on detecting vulnerabilities in Solidity smart contracts using Machine Learning and Deep Learning techniques. It is part of an InfoSec project topic and leverages the BCCC-VulSCs-2023 dataset for comprehensive analysis and model training.

## Dataset

The **BCCC-VulSCs-2023 Dataset** is a substantial collection for Solidity Smart Contracts (SCs) analysis, featuring:

- **36,670 samples** of Solidity smart contracts
- **70 feature columns** enriched with vulnerability and contract characteristics
- Comprehensive data for machine learning and deep learning applications

## Technologies & Libraries

- **Python**: Core programming language
- **Jupyter Notebook**: Interactive development and analysis
- **Machine Learning Libraries**: scikit-learn, XGBoost, and others
- **Deep Learning Framework**: TensorFlow/Keras or PyTorch
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries (see requirements.txt or environment specifications)

### Installation

```bash
# Clone the repository
git clone https://github.com/Kenleo14/Vulnerable-Smart-Contracts-Detection-using-ML-DL.git

# Navigate to the project directory
cd Vulnerable-Smart-Contracts-Detection-using-ML-DL

# Install dependencies
pip install -r requirements.txt

# run cleanup script
python clean_bccc_vuls_dataset.py

# run main train model script
python main_bert_vuln_classification.py
or
main_bert_vuln_classification.ipynb
