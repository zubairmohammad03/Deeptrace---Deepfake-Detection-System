#!/usr/bin/env python
# ml/training/train.py
"""
DeepTrace AI — Training Script

Usage:
    python train.py
    python train.py --epochs 20 --batch_size 64 --backbone efficientnet_b4
    python train.py --arch dual --epochs 30
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm

# Add ml/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.deepfake_dataset import get_dataloaders
from models.deepfake_model import build_model, count_params

# ── Config ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepTrace Training")
    p.add_argument("--data_root",     default="data",             help="Path to dataset root")
    p.add_argument("--arch",          default="single",           choices=["single", "dual"])
    p.add_argument("--backbone",      default="efficientnet_b4",  help="timm model name")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--epochs",        type=int, default=20)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--max_per_source",type=int, default=5000)
    p.add_argument("--output_dir",    default="checkpoints")
    p.add_argument("--workers",       type=int, default=4)
    p.add_argument("--device",        default="auto")
    return p.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / total, correct / total, auc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*55}")
    print(f"  DeepTrace AI — Training")
    print(f"{'='*55}")
    print(f"  Device   : {device}")
    print(f"  Backbone : {args.backbone}")
    print(f"  Arch     : {args.arch}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"{'='*55}\n")

    # Data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_per_source=args.max_per_source,
    )

    # Model
    model = build_model(architecture=args.arch, backbone=args.backbone).to(device)
    print(f"Model params: {count_params(model)}\n")

    # Optimizer / loss / scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "val_auc": round(val_auc, 4)
        })

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "args": vars(args)
            }, ckpt_path)
            print(f"  ✓ Saved best model (AUC={val_auc:.4f}) → {ckpt_path}")

    # Save history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training complete! Best Val AUC: {best_auc:.4f}")
    print(f"  Checkpoints saved to: {out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
