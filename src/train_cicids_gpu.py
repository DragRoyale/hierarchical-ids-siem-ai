import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. SETTINGS
# =========================
DATA_DIR = Path(r"C:\Users\Dragroyale\Desktop\content\vscode\siem-ai-q1\data\raw\CICIDS2017") 
BATCH_SIZE = 4096
EPOCHS = 15
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_WORKERS = 0   # on Windows keep 0 if problems
MODEL_SAVE_PATH = "cicids_binary_mlp.pth"


# =========================
# 2. REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)


# =========================
# 3. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 4. LOAD ALL CSV FILES
# =========================
csv_files = list(DATA_DIR.glob("*.csv"))
print("Found files:", len(csv_files))

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

dfs = []
for file in csv_files:
    print("Loading:", file.name)
    df = pd.read_csv(file, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print("Raw shape:", data.shape)


# =========================
# 5. CLEAN COLUMN NAMES
# =========================
data.columns = data.columns.str.strip()
print("\nColumns after strip:")
print(data.columns.tolist())

if "Label" not in data.columns:
    raise KeyError("Column 'Label' not found after cleaning column names.")

print("\nLabel distribution before cleaning:")
print(data["Label"].value_counts())


# =========================
# 6. CREATE BINARY LABEL
# =========================
data["binary_label"] = (data["Label"] != "BENIGN").astype(np.int64)

print("\nBinary label distribution:")
print(data["binary_label"].value_counts())


# =========================
# 7. KEEP FEATURES + LABELS
# =========================
feature_cols = [col for col in data.columns if col not in ["Label", "binary_label"]]
X = data[feature_cols].copy()
y = data["binary_label"].copy()

print("\nInitial X shape:", X.shape)
print("Initial y shape:", y.shape)


# =========================
# 8. FORCE NUMERIC
# =========================
non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
print("\nNon-numeric columns:", non_numeric)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# =========================
# 9. REMOVE INF / NAN
# =========================
X = X.replace([np.inf, -np.inf], np.nan)

valid_mask = X.notna().all(axis=1)
X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()

print("\nShape after removing NaN rows:")
print("X:", X.shape, "y:", y.shape)


# =========================
# 10. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

print("\nTrain label distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest label distribution:")
print(y_test.value_counts(normalize=True))


# =========================
# 11. STANDARDIZATION
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)

print("\nScaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)


# =========================
# 12. DATASET / DATALOADER
# =========================
class CICIDSDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CICIDSDataset(X_train_scaled, y_train_np)
test_dataset = CICIDSDataset(X_test_scaled, y_test_np)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# =========================
# 13. MODEL
# =========================
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


input_dim = X_train_scaled.shape[1]
model = MLPBinaryClassifier(input_dim).to(device)
print("\nModel created with input_dim =", input_dim)


# =========================
# 14. LOSS WITH CLASS WEIGHT
# =========================
num_neg = (y_train_np == 0).sum()
num_pos = (y_train_np == 1).sum()

pos_weight_value = num_neg / max(num_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("pos_weight =", pos_weight_value)


# =========================
# 15. TRAINING LOOP
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    all_logits = []
    all_labels = []

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        logits = model(batch_X)

        all_logits.append(logits.cpu())
        all_labels.append(batch_y)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro")
    roc_auc = roc_auc_score(all_labels, probs)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "labels": all_labels.astype(int),
        "preds": preds,
        "probs": probs,
    }


best_f1 = -1.0
history = []

print("\nStarting training...\n")
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    metrics = evaluate(model, test_loader, device, threshold=0.5)

    elapsed = time.time() - start_time

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_roc_auc": metrics["roc_auc"],
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if metrics["macro_f1"] > best_f1:
        best_f1 = metrics["macro_f1"]
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Best model saved to {MODEL_SAVE_PATH}")


# =========================
# 16. LOAD BEST MODEL
# =========================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
final_metrics = evaluate(model, test_loader, device, threshold=0.5)

y_true = final_metrics["labels"]
y_pred = final_metrics["preds"]
y_prob = final_metrics["probs"]

print("\nFinal Test Metrics")
print("Accuracy :", round(final_metrics["accuracy"], 4))
print("Macro-F1 :", round(final_metrics["macro_f1"], 4))
print("ROC-AUC  :", round(final_metrics["roc_auc"], 4))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))


# =========================
# 17. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
disp.plot()
plt.title("CICIDS2017 Binary Classification - Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================
# 18. ROC CURVE
# =========================
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CICIDS2017 Binary Classification - ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 19. THROUGHPUT
# =========================
@torch.no_grad()
def measure_throughput(model, X_data, device, batch_size=8192):
    model.eval()
    dataset = torch.tensor(X_data, dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start = time.time()
    total = 0

    for batch_X in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        _ = model(batch_X)
        total += batch_X.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start
    samples_per_sec = total / elapsed
    return samples_per_sec, elapsed


samples_per_sec, elapsed = measure_throughput(model, X_test_scaled, device)
print(f"\nInference throughput: {samples_per_sec:.2f} samples/sec")
print(f"Measured on {len(X_test_scaled)} test samples in {elapsed:.2f} sec")


# =========================
# 20. TRAINING CURVES
# =========================
history_df = pd.DataFrame(history)

plt.figure(figsize=(7, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history_df["epoch"], history_df["test_macro_f1"], marker="o", label="Macro-F1")
plt.plot(history_df["epoch"], history_df["test_accuracy"], marker="o", label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Test Metrics by Epoch")
plt.legend()
plt.tight_layout()
plt.show()