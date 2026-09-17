# ml/datasets/deepfake_dataset.py
"""
Dataset loaders for deepfake detection training.

Supports:
  - FaceForensics++ (FF++)
  - Celeb-DF v2
  - DFDC (DeepFake Detection Challenge)
  - WildDeepfake

Directory structure expected:
    data/
      faceforensics/
        real/     *.jpg / *.png
        fake/     *.jpg / *.png
      celebdf/
        real/
        fake/
      dfdc/
        real/
        fake/
      wilddeepfake/
        real/
        fake/
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.ToGray(p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Core Dataset ─────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Unified dataset for deepfake detection across multiple sources.

    Args:
        root_dirs: list of (real_dir, fake_dir) tuples
        transform: albumentations Compose
        max_per_source: cap samples per source for balance
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root_dirs: List[Tuple[str, str]],
        transform=None,
        max_per_source: Optional[int] = None,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (path, label)  0=real, 1=fake

        for real_dir, fake_dir in root_dirs:
            reals = self._collect(real_dir, label=0, cap=max_per_source)
            fakes = self._collect(fake_dir, label=1, cap=max_per_source)
            self.samples.extend(reals + fakes)

        random.shuffle(self.samples)
        print(f"[Dataset] Total: {len(self.samples)} samples "
              f"(real={sum(1 for _,l in self.samples if l==0)}, "
              f"fake={sum(1 for _,l in self.samples if l==1)})")

    def _collect(self, directory: str, label: int, cap: Optional[int]) -> List[Tuple[str, int]]:
        p = Path(directory)
        if not p.exists():
            print(f"[Dataset] Warning: {directory} not found, skipping")
            return []
        files = [str(f) for f in p.rglob("*") if f.suffix.lower() in self.EXTENSIONS]
        if cap:
            files = files[:cap]
        return [(f, label) for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Fallback: black image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)


# ── Weighted sampler for class imbalance ─────────────────────────────────────

def make_weighted_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    labels = [s[1] for s in dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: str = "data",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    max_per_source: int = 5000,
):
    """
    Build train/val DataLoaders from all available datasets.
    """
    sources = []
    for name in ["faceforensics", "celebdf", "dfdc", "wilddeepfake"]:
        real = os.path.join(data_root, name, "real")
        fake = os.path.join(data_root, name, "fake")
        if os.path.exists(real) and os.path.exists(fake):
            sources.append((real, fake))

    if not sources:
        raise FileNotFoundError(
            f"No dataset directories found under {data_root}/. "
            "Please download the datasets (see ml/datasets/DOWNLOAD.md)"
        )

    full_ds = DeepfakeDataset(sources, transform=None, max_per_source=max_per_source)

    # Split
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    # Apply transforms
    train_ds.dataset.transform = get_train_transforms(img_size)
    val_ds.dataset.transform = get_val_transforms(img_size)

    sampler = make_weighted_sampler(full_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
