import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import nn
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib. pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import Counter

# CONFIGURATION
sns.set(style="whitegrid")

MODEL_NAME = "microsoft/codebert-base"
MODEL_PATH = "codebert_contract_vuln_classifier.pt"
TEST_CSV = "full_contracts_dataset_with_source.csv"  # Or your new test set
MAX_LEN = 256
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization sampling to avoid heavy computation on very large datasets
N_SAMPLES_FOR_VIZ = 3000  # adjust as needed
TOP_TOKENS_N = 25

LABEL_NAMES = {0: "Secure (0)", 1: "Vulnerable (1)"}

print(f"Using device: {DEVICE}")

# LOAD AND PREPARE TEST DATA
def load_test_data(test_csv_path):
    """Load and prepare test data from CSV."""
    test_path = Path(test_csv_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    df = pd.read_csv(test_csv_path)
    df = df[["source_code", "label"]].dropna()
    df = df[df["source_code"].str.len() > 0]

    # Ensure labels are integers 0/1
    df["label"] = df["label"].astype(int)
    print(f"Test samples: {len(df)}")
    return df

# DATASET VISUALIZATION
def visualize_class_distribution(df):
    """Plot class distribution."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="label", data=df, palette="Set2")
    ax.set_xticklabels([LABEL_NAMES. get(int(x.get_text()), str(x.get_text())) 
                        for x in ax. get_xticklabels()])
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def visualize_character_lengths(df):
    """Plot character length distributions."""
    df["char_len"] = df["source_code"]. str.len()

    plt.figure(figsize=(10, 4))
    sns.histplot(df["char_len"], bins=50, kde=False, color="tab:blue")
    plt.title("Character Length Distribution (All)")
    plt.xlabel("Characters per snippet")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="char_len", hue="label", bins=50, kde=False, 
                 palette="Set2", stat="count")
    plt.title("Character Length Distribution by Class")
    plt.xlabel("Characters per snippet")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="label", y="char_len", palette="Set2")
    plt.title("Character Length by Class (Boxplot)")
    plt. xlabel("Class")
    plt.ylabel("Characters per snippet")
    plt.tight_layout()
    plt.show()

def visualize_token_lengths(df, tokenizer, max_len=MAX_LEN, n_samples=N_SAMPLES_FOR_VIZ):
    """Plot token length distributions using CodeBERT tokenizer."""
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42). copy()

    def token_len(text, max_len=max_len):
        enc = tokenizer(
            str(text)[:5000],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        return int(enc["input_ids"].shape[1])

    tqdm.pandas(desc="Tokenizing for length")
    sample_df["token_len"] = sample_df["source_code"].progress_apply(
        lambda s: token_len(s, max_len)
    )

    plt.figure(figsize=(10, 4))
    sns.histplot(sample_df["token_len"], bins=max_len//8, kde=False, 
                 color="tab:purple")
    plt.title("Token Length Distribution (Sampled)")
    plt.xlabel("Tokens per snippet (padded/truncated to max_length)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns. violinplot(data=sample_df, x="label", y="token_len", palette="Set2", cut=0)
    plt.title("Token Length by Class (Violin, Sampled)")
    plt.xlabel("Class")
    plt.ylabel("Tokens (padded/truncated)")
    plt.tight_layout()
    plt.show()

def check_duplicates(df):
    """Check for duplicate source code entries."""
    dup_mask = df.duplicated(subset=["source_code"], keep=False)
    num_dups = dup_mask.sum()
    num_unique = df.shape[0] - df["source_code"].nunique()
    
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows (any repeated source_code): {num_dups}")
    print(f"Rows part of duplicated groups (len(df) - nunique): {num_unique}")

    if num_dups > 0:
        dup_examples = (df[dup_mask]
                       .groupby("source_code")
                       .head(2)
                       .reset_index(drop=True)
                       . head(6))
        print("\nSample duplicate entries:")
        print(dup_examples)
    else:
        print("No duplicate source_code entries detected.")

def visualize_top_tokens(df, tokenizer, max_len=MAX_LEN, 
                         n_samples=N_SAMPLES_FOR_VIZ, top_n=TOP_TOKENS_N):
    """Visualize top token frequencies (sampled)."""
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42). copy()
    
    special_tokens = set([
        tokenizer.cls_token, 
        tokenizer.sep_token, 
        tokenizer.pad_token, 
        tokenizer.unk_token
    ])
    token_counter = Counter()

    for text in tqdm(sample_df["source_code"]. tolist(), desc="Counting tokens"):
        enc = tokenizer(
            str(text)[:5000],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0). tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        for t in toks:
            if t not in special_tokens:
                token_counter[t] += 1

    top_tokens = token_counter.most_common(top_n)
    top_df = pd.DataFrame(top_tokens, columns=["token", "count"])

    plt.figure(figsize=(12, 6))
    sns. barplot(data=top_df, x="count", y="token", palette="viridis")
    plt.title(f"Top {top_n} Token Frequencies (Sampled)")
    plt.xlabel("Count")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} tokens:")
    print(top_df.head(10))

# DATASET AND DATALOADER
class ContractDataset(Dataset):
    """Dataset for smart contract source code and vulnerability labels."""
    
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["source_code"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        code = str(self.texts[idx])[:5000]
        enc = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_ids, attn_mask, label

# MODEL DEFINITION
class CodeBERTClassifier(nn. Module):
    """CodeBERT-based classifier for vulnerability detection."""
    
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self. fc = nn.Linear(self. backbone.config.hidden_size, 1)

    def forward(self, input_ids, attn_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attn_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_rep)
        logits = self.fc(x)
        return logits

# EVALUATION
def evaluate_model(model, dataloader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    all_preds = []
    all_labels = []
    all_losses = []
    batch_accuracies = []
    batch_losses = []

    with torch.no_grad():
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            input_ids = input_ids. to(device)
            attn_mask = attn_mask. to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits. squeeze(-1), labels)

            all_losses.append(loss.item())
            batch_losses.append(loss.item())

            preds = torch.sigmoid(logits).squeeze(-1) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_acc = accuracy_score(labels.cpu().numpy(), preds. cpu().numpy())
            batch_accuracies.append(batch_acc)

    overall_accuracy = accuracy_score(all_labels, all_preds)
    average_loss = np.mean(all_losses)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Average Loss (BCEWithLogits): {average_loss:.4f}")

    cls_report = classification_report(all_labels, all_preds)
    print(cls_report)

    return {
        "accuracy": overall_accuracy,
        "loss": average_loss,
        "predictions": all_preds,
        "labels": all_labels,
        "batch_accuracies": batch_accuracies,
        "batch_losses": batch_losses,
        "classification_report": cls_report,
    }

def plot_evaluation_metrics(eval_results):
    """Plot accuracy and loss over batches."""
    batch_accs = eval_results["batch_accuracies"]
    batch_losses = eval_results["batch_losses"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(batch_accs, alpha=0.7, color="green")
    ax1.axhline(y=eval_results["accuracy"], color="r", linestyle="--", 
                label=f"Overall: {eval_results['accuracy']:. 4f}")
    ax1. set_xlabel("Batch Index")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Batch-wise Accuracy")
    ax1.legend()
    ax1. grid(True)

    ax2.plot(batch_losses, alpha=0.7, color="blue")
    ax2.axhline(y=eval_results["loss"], color="r", linestyle="--", 
                label=f"Average: {eval_results['loss']:.4f}")
    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("Loss (BCEWithLogits)")
    ax2.set_title("Batch-wise Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# MAIN EXECUTION
def main():
    """Main evaluation pipeline."""
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    df = load_test_data(TEST_CSV)

    # Load tokenizer
    print("\n" + "="*80)
    print("LOADING TOKENIZER")
    print("="*80)
    tokenizer = AutoTokenizer. from_pretrained(MODEL_NAME)

    # Visualize dataset
    print("\n" + "="*80)
    print("DATASET VISUALIZATION")
    print("="*80)
    
    visualize_class_distribution(df)
    visualize_character_lengths(df)
    visualize_token_lengths(df, tokenizer)
    check_duplicates(df)
    visualize_top_tokens(df, tokenizer)

    # Create dataset and dataloader
    print("\n" + "="*80)
    print("CREATING DATASET AND DATALOADER")
    print("="*80)
    test_ds = ContractDataset(df, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    print(f"Dataset size: {len(test_ds)}")

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = CodeBERTClassifier(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded and set to eval mode.")

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    eval_results = evaluate_model(model, test_loader, DEVICE)

    # Plot metrics
    print("\n" + "="*80)
    print("PLOTTING EVALUATION METRICS")
    print("="*80)
    plot_evaluation_metrics(eval_results)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()