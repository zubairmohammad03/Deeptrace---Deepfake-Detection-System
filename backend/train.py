"""
DeepTrace AI — Model Training Script
Train EfficientNet-B4 on deepfake detection datasets.

Datasets expected (download separately):
  - FaceForensics++ (FF++): https://github.com/ondyari/FaceForensics
  - DFDC: https://ai.facebook.com/datasets/dfdc/
  - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
  - WildDeepfake: https://github.com/deepfakeinthewild/deepfake-in-the-wild

Usage:
    python train.py --dataset_root /data/deepfakes --model efficientnet --epochs 30

Data directory structure expected:
    /data/deepfakes/
        train/
            real/    ← real face images
            fake/    ← deepfake face images
        val/
            real/
            fake/
        test/
            real/
            fake/
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Binary classification dataset.
    Loads real (label=0) and fake (label=1) face images.
    Applies augmentation to reduce overfitting and improve generalization.
    """
    AUGMENT = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    EVAL = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, root: str, split: str = "train"):
        self.transform = self.AUGMENT if split == "train" else self.EVAL
        self.samples = []
        for label, cls in enumerate(["real", "fake"]):
            folder = Path(root) / split / cls
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for path in folder.glob(ext):
                    self.samples.append((str(path), label))
        logger.info(f"Dataset [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(args.dataset_root, "train")
    val_ds   = DeepfakeDataset(args.dataset_root, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "efficientnet":
        model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=1, drop_rate=0.3)
        checkpoint_name = "efficientnet_b4_deepfake.pth"
    elif args.model == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)
        checkpoint_name = "vit_b16_deepfake.pth"
    elif args.model == "xception":
        model = timm.create_model("xception", pretrained=True, num_classes=1)
        checkpoint_name = "xception_deepfake.pth"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # ── Loss: Binary Cross-Entropy with label smoothing ───────────────────────
    # Label smoothing 0.1 prevents overconfidence and improves calibration.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # ── Optimizer: AdamW + Cosine LR schedule ─────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0

    for epoch in range(args.epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()

            # Mixup augmentation (helps generalization across datasets)
            if args.mixup and torch.rand(1).item() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(images.size(0))
                images = lam * images + (1 - lam) * images[idx]
                labels = lam * labels + (1 - lam) * labels[idx]

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = torch.sigmoid(model(images))
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        auc = roc_auc_score(all_labels, all_probs)
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)

        logger.info(f"Epoch {epoch+1} | Val AUC: {auc:.4f} | Val Acc: {acc:.4f} | Train Loss: {train_loss/len(train_loader):.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{checkpoint_name}")
            logger.info(f"  ✅ New best! Saved checkpoint: checkpoints/{checkpoint_name}")

    logger.info(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    logger.info(f"Set EFFICIENTNET_CHECKPOINT=checkpoints/{checkpoint_name} in .env to use this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/deepfakes")
    parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "vit", "xception"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixup", action="store_true", default=True)
    train(parser.parse_args())
