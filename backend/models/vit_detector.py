"""
DeepTrace AI — Vision Transformer (ViT) Deepfake Detector
Captures long-range spatial inconsistencies that CNNs miss.

Architecture:
  - Backbone: ViT-Base/16 (16×16 patch size)
  - Patches: 224×224 image → 196 patches of size 16×16
  - Classification token [CLS] used for binary prediction
  - Attention heads: 12 heads × 12 layers
  - Embedding dim: 768

Why ViT for deepfakes?
  CNNs are great at local texture artifacts, but deepfakes often have
  GLOBAL inconsistencies — e.g., the lighting on the face doesn't match
  the background, or the ear doesn't match the jawline across the image.
  ViT's self-attention mechanism naturally captures these global correlations.

Key insight: Attention rollout maps show WHICH patches attend to which,
revealing unnatural attention patterns in manipulated faces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding space."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)              # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with attention map storage for XAI."""

    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.attention_weights = None  # Store for attention rollout

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        self.attention_weights = attn.detach()  # (B, heads, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDetector(nn.Module):
    """
    Full Vision Transformer for deepfake detection.
    Uses attention rollout for XAI visualization.
    """

    def __init__(
        self,
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=768, depth=12, n_heads=12,
        mlp_ratio=4.0, dropout=0.1, pretrained=True
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Binary classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self._init_weights()

        if pretrained:
            self._load_pretrained()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _load_pretrained(self):
        """Attempt to load timm pretrained ViT weights."""
        try:
            import timm
            vit = timm.create_model("vit_base_patch16_224", pretrained=True)
            # Copy backbone weights, skip final head
            self.load_state_dict(vit.state_dict(), strict=False)
        except Exception:
            pass  # Use random init if timm not available

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        # Patch embedding + positional encoding
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token → classification

        logits = self.head(cls_out)
        return logits, cls_out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    def get_attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention Rollout for ViT XAI.
        
        Algorithm (Abnar & Zuidema, 2020):
          1. Collect attention maps from all layers
          2. Add identity matrix (residual connections)
          3. Multiply across layers (rollout)
          4. Extract [CLS] → patch attentions
          5. Reshape to 2D spatial map

        Returns:
            heatmap: (14, 14) attention rollout map
        """
        self.eval()
        with torch.no_grad():
            self.forward(x)

        n_patches_side = int(math.sqrt(self.patch_embed.n_patches))  # 14 for 224×224
        rollout = torch.eye(self.patch_embed.n_patches + 1)

        for block in self.blocks:
            attn = block.attn.attention_weights  # (B, heads, N+1, N+1)
            if attn is None:
                continue
            # Average over heads, add identity (residual)
            attn_avg = attn[0].mean(0)
            attn_avg = attn_avg + torch.eye(attn_avg.shape[0])
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout = torch.mm(attn_avg, rollout)

        # CLS token attention to all patches
        cls_attn = rollout[0, 1:]  # skip CLS self-attention
        heatmap = cls_attn.reshape(n_patches_side, n_patches_side)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "ViTDetector":
        model = cls(pretrained=False)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(device)
        model.eval()
        return model
