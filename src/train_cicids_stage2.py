import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path(r"C:\Users\Dragroyale\Desktop\content\vscode\siem-ai-q1\data\raw\CICIDS2017")   
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_WORKERS = 0
MODEL_SAVE_PATH = "cicids_stage2_multiclass.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")


csv_files = list(DATA_DIR.glob("*.csv"))
print("Found files:", len(csv_files))

dfs = []
for file in csv_files:
    print("Loading:", file.name)
    df = pd.read_csv(file, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip()

print("Raw shape:", data.shape)


data["Label"] = data["Label"].astype(str).str.strip()

label_map = {
    "Web Attack – Brute Force": "Web Attack Brute Force",
    "Web Attack – XSS": "Web Attack XSS",
    "Web Attack – Sql Injection": "Web Attack Sql Injection",
    "Web Attack - Brute Force": "Web Attack Brute Force",
    "Web Attack - XSS": "Web Attack XSS",
    "Web Attack - Sql Injection": "Web Attack Sql Injection",
    "Web Attack � Brute Force": "Web Attack Brute Force",
    "Web Attack � XSS": "Web Attack XSS",
    "Web Attack � Sql Injection": "Web Attack Sql Injection",
}
data["Label"] = data["Label"].replace(label_map)

print("\nLabel counts:")
print(data["Label"].value_counts())


attack_data = data[data["Label"] != "BENIGN"].copy()

print("\nAttack-only shape:", attack_data.shape)
print("\nAttack label counts:")
print(attack_data["Label"].value_counts())


feature_cols = [col for col in attack_data.columns if col != "Label"]

X = attack_data[feature_cols].copy()
y_text = attack_data["Label"].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

valid_mask = X.notna().all(axis=1)
X = X.loc[valid_mask].copy()
y_text = y_text.loc[valid_mask].copy()

print("\nShape after cleaning:")
print("X:", X.shape)
print("y:", y_text.shape)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

class_names = label_encoder.classes_
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.astype(np.int64)
y_test_np = y_test.astype(np.int64)


class AttackDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = AttackDataset(X_train_scaled, y_train_np)
test_dataset = AttackDataset(X_test_scaled, y_test_np)

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


class MLPStage2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_scaled.shape[1]
model = MLPStage2(input_dim, num_classes).to(device)
print("\nModel device:", next(model.parameters()).device)


class_counts = np.bincount(y_train_np)
class_weights = len(y_train_np) / (num_classes * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


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

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_X)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(batch_y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, macro_f1, all_labels, all_preds


best_f1 = -1.0
history = []

print("\nStarting Stage 2 training...\n")
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    acc, macro_f1, _, _ = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Best model saved to {MODEL_SAVE_PATH}")


model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
final_acc, final_macro_f1, y_true, y_pred = evaluate(model, test_loader, device)

print("\nFinal Stage 2 Metrics")
print("Accuracy :", round(final_acc, 4))
print("Macro-F1 :", round(final_macro_f1, 4))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(cm_norm)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("CICIDS2017 Stage 2 - Normalized Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        value = cm_norm[i, j]
        if value > 0.01:  
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.colorbar(im)
plt.tight_layout()
plt.show()


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
print(f"\nStage 2 inference throughput: {samples_per_sec:.2f} samples/sec")
print(f"Measured on {len(X_test_scaled)} samples in {elapsed:.2f} sec")
