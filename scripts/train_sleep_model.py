#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import Models
from models.cnn_model import Simple1DCNN
from models.conv_lstm_model import ConvLSTMModel
from models.transformer_model import Transformer1DModel

# Metrics function
from utils import per_class_metrics

# Dataset Class
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])

# Training
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)

# Evaluation
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)

# Confusion Matrix Plot
def save_confusion_matrix(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), va="center", ha="center")

    plt.colorbar()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

# Main training pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="Dataset/sleep_windows.npz")
    parser.add_argument("-labels", default="Dataset/sleep_labels.csv")
    parser.add_argument("-model", choices=["cnn", "conv_lstm", "transformer"], default="cnn")
    parser.add_argument("-device", default="mps")
    args = parser.parse_args()

    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load data
    data = np.load(args.dataset)
    X_raw, y_raw = data["X"], data["y"]

    meta = pd.read_csv(args.labels)
    participants = meta["participant"].values

    unique_classes = sorted(list(set(y_raw)))

    mapping_file = "Dataset/sleep_label_mapping.txt"
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            idx, label = line.strip().split("\t")
            mapping[int(idx)] = label

    class_names = [mapping[c] for c in unique_classes]

    print("Classes:", class_names)

    dataset = WindowDataset(X_raw, y_raw)
    folds = np.unique(participants)
    all_metrics = []

    os.makedirs("sleep_results", exist_ok=True)

    # LOSO CV
    for f in folds:
        print(f"\n=== Fold test participant = {f} ===")

        test_idx = np.where(participants == f)[0]
        train_idx = np.where(participants != f)[0]

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=64, shuffle=False)

        # Select model
        if args.model == "cnn":
            model = Simple1DCNN()
        elif args.model == "conv_lstm":
            model = ConvLSTMModel()
        else:
            model = Transformer1DModel()

        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train 20 epochs
        for e in range(20):
            train_epoch(model, train_loader, optim, loss_fn, device)

        # Evaluate
        trues, preds = evaluate(model, test_loader, device)
        metrics = per_class_metrics(trues, preds, labels=unique_classes)
        all_metrics.append(metrics)

        # Save confusion matrix
        cm_path = f"sleep_results/confusion_{args.model}_AP{f}.png"
        save_confusion_matrix(trues, preds, class_names, cm_path)

        print("Saved confusion matrix:", cm_path)
        print("Metrics:", metrics)

    # Save all metrics
    outcsv = f"sleep_results/results_{args.model}_sleep_metrics.csv"
    df = pd.DataFrame(all_metrics)
    df.to_csv(outcsv, index=False)

    print("\nAll folds complete.")
    print("Saved metrics:", outcsv)


if __name__ == "__main__":
    main()
