import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import nn
from sklearn.metrics import classification_report
from tqdm import tqdm

MODEL_NAME = "microsoft/codebert-base"
MODEL_PATH = "codebert_contract_vuln_classifier.pt"
TEST_CSV = "full_contracts_dataset_with_source.csv"  # Or your new test set
MAX_LEN = 256
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare test data
df = pd.read_csv(TEST_CSV)
df = df[["source_code", "label"]].dropna()
df = df[df["source_code"].str.len() > 0]

class ContractDataset(Dataset):
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_ds = ContractDataset(df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Define model (must match your train script)
class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attn_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attn_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_rep)
        logits = self.fc(x)
        return logits

# Load model weights
model = CodeBERTClassifier(MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Inference
preds, trues = [], []
with torch.no_grad():
    for input_ids, attn_mask, labels in tqdm(test_loader):
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        logits = model(input_ids, attn_mask).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend((probs > 0.5).astype(int))
        trues.extend(labels.numpy())
print(classification_report(trues, preds, digits=4))