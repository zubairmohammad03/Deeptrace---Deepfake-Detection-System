#!/usr/bin/env python
# ml/training/evaluate.py
"""
Evaluate a trained DeepTrace model with full metrics report.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_root data
"""
import sys, argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.deepfake_dataset import get_dataloaders
from models.deepfake_model import build_model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    model = build_model(
        architecture=saved_args.get("arch", "single"),
        backbone=saved_args.get("backbone", "efficientnet_b4"),
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=64,
        num_workers=4,
    )

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print("  DeepTrace — Evaluation Report")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=['Real','Fake'])}")
    print("="*50)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#00f5a0", lw=2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend()

    axes[1].imshow(cm, cmap="Blues")
    axes[1].set(xticks=[0,1], yticks=[0,1], xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual"); axes[1].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)

    plt.tight_layout()
    out = Path(args.checkpoint).parent / "evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plots saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--data_root", default="data")
    evaluate(p.parse_args())
