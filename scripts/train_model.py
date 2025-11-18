#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from models.cnn_model import Simple1DCNN
from models.conv_lstm_model import ConvLSTMModel
from utils import per_class_metrics

LABEL_ORDER = ["Hypopnea", "Obstructive Apnea", "Normal"]


# Dataset
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])


# Training
def train_epoch(model, loader, loss_fn, optim, device):
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
def eval_model(model, loader, device):
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
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

# Metrics Printer
def print_metrics(all_fold_metrics):
    print("\n=== METRICS SUMMARY ===\n")

    for i, m in enumerate(all_fold_metrics):
        print(f"--- Fold {i+1} ---")
        print("Confusion Matrix:\n", m["confusion_matrix"])
        print("Precision:", np.round(m["precision"], 3))
        print("Recall:", np.round(m["recall"], 3))
        print("Specificity:", np.round(m["specificity"], 3))
        print("Sensitivity:", np.round(m["recall"], 3))  # same as recall
        print("Accuracy:", np.round(m["accuracy"], 3))
        print()

    mean_precision = np.mean([m["precision"] for m in all_fold_metrics], axis=0)
    mean_recall = np.mean([m["recall"] for m in all_fold_metrics], axis=0)
    mean_specific = np.mean([m["specificity"] for m in all_fold_metrics], axis=0)

    print("=== AVERAGE OVER ALL FOLDS ===")
    print("Precision:", np.round(mean_precision, 3))
    print("Recall:", np.round(mean_recall, 3))
    print("Specificity:", np.round(mean_specific, 3))

# Main (LOSO CV)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="Dataset/breathing_windows.npz")
    parser.add_argument("-meta", default="Dataset/breathing_labels.csv")
    parser.add_argument("-model", choices=["cnn", "conv_lstm"], default="cnn")
    parser.add_argument("-device", default="mps")
    args = parser.parse_args()

    # Select device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load data
    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    meta = pd.read_csv(args.meta)
    participants = meta["participant"].values

    sets = np.unique(participants)
    all_fold_metrics = []

    os.makedirs("breathing_results", exist_ok=True)

    for test_p in sets:
        print(f"\n=== Fold test participant = {test_p} ===")
        test_idx = np.where(participants == test_p)[0]
        train_idx = np.where(participants != test_p)[0]

        ds = WindowDataset(X, y)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False)

        # Choose model
        if args.model == "cnn":
            model = Simple1DCNN()
        else:
            model = ConvLSTMModel()

        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        for _ in range(20):
            train_epoch(model, train_loader, loss_fn, optim, device)

        # Evaluate
        trues, preds = eval_model(model, test_loader, device)
        metrics = per_class_metrics(trues, preds, labels=[0, 1, 2])
        all_fold_metrics.append(metrics)

        # Save confusion matrix
        save_confusion_matrix(
            trues, preds, LABEL_ORDER,
            f"breathing_results/confusion_{args.model}_AP{test_p}.png"
        )

        print(f"Saved confusion matrix → breathing_results/confusion_{args.model}_AP{test_p}.png")

    print_metrics(all_fold_metrics)

    # Save CSV
    csv_path = f"breathing_results/results_{args.model}_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("Fold,Class,Precision,Recall,Sensitivity,Specificity,Accuracy\n")
        for fold_idx, m in enumerate(all_fold_metrics):
            for c in range(3):
                f.write(f"{fold_idx+1},{LABEL_ORDER[c]},"
                        f"{m['precision'][c]},"
                        f"{m['recall'][c]},"
                        f"{m['recall'][c]},"
                        f"{m['specificity'][c]},"
                        f"{m['accuracy'][c]}\n")

    print(f"\nAll results saved to: {csv_path}")
    print("Complete!")


if __name__ == "__main__":
    main()
