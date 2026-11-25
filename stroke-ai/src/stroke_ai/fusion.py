"""Networks (image-only, feature-level fusion, cross-modal attention) and late fusion.

All networks share the signature  net(img, clin) -> logit  of shape (B,). Image-only nets ignore `clin`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .imaging_model import make_backbone


class ImageOnlyNet(nn.Module):
    def __init__(self, backbone: str = "small_cnn", embed_dim: int = 64, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.backbone = make_backbone(backbone, pretrained)
        self.proj = nn.Sequential(nn.Linear(self.backbone.out_channels, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, img, clin=None):
        f = self.backbone(img).mean(dim=(2, 3))
        return self.head(self.proj(f)).squeeze(1)


class FeatureFusionNet(nn.Module):
    """Image embedding || clinical embedding -> MLP. Trained end to end."""

    def __init__(self, clin_dim: int, backbone: str = "small_cnn", embed_dim: int = 64, dropout: float = 0.3,
                 pretrained: bool = False):
        super().__init__()
        self.backbone = make_backbone(backbone, pretrained)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_channels, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.clin_enc = nn.Sequential(
            nn.Linear(clin_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, img, clin):
        fi = self.img_proj(self.backbone(img).mean(dim=(2, 3)))
        fc = self.clin_enc(clin)
        return self.head(torch.cat([fi, fc], dim=1)).squeeze(1)


class CrossAttentionFusionNet(nn.Module):
    """Each clinical variable becomes a token (value * learned vector + bias); the image feature map becomes
    spatial tokens. Clinical tokens attend to image tokens and vice versa (bidirectional cross-attention),
    each with a residual connection + LayerNorm, then mean-pooled and classified."""

    def __init__(self, clin_dim: int, backbone: str = "small_cnn", d_model: int = 64, n_heads: int = 4,
                 dropout: float = 0.2, pretrained: bool = False):
        super().__init__()
        self.backbone = make_backbone(backbone, pretrained)
        self.img_tok = nn.Linear(self.backbone.out_channels, d_model)
        self.clin_w = nn.Parameter(torch.randn(clin_dim, d_model) * 0.02)
        self.clin_b = nn.Parameter(torch.zeros(clin_dim, d_model))
        self.attn_c2i = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_i2c = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_i = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, img, clin):
        fm = self.backbone(img)                                  # B, C, h, w
        it = self.img_tok(fm.flatten(2).transpose(1, 2))         # B, h*w, d
        ct = clin.unsqueeze(-1) * self.clin_w + self.clin_b      # B, F, d
        c_att, _ = self.attn_c2i(ct, it, it)
        i_att, _ = self.attn_i2c(it, ct, ct)
        c = self.norm_c(ct + c_att)
        i = self.norm_i(it + i_att)
        return self.head(torch.cat([c.mean(1), i.mean(1)], dim=1)).squeeze(1)


NET_KINDS = ("image_only", "feature_fusion", "cross_attention")


def build_net(kind: str, clin_dim: int = 1, backbone: str = "small_cnn", embed_dim: int = 64,
              pretrained: bool = False) -> nn.Module:
    if kind == "image_only":
        net = ImageOnlyNet(backbone, embed_dim, pretrained=pretrained)
    elif kind == "feature_fusion":
        net = FeatureFusionNet(clin_dim, backbone, embed_dim, pretrained=pretrained)
    elif kind == "cross_attention":
        net = CrossAttentionFusionNet(clin_dim, backbone, embed_dim, pretrained=pretrained)
    else:
        raise ValueError(f"unknown net kind {kind!r}; choose from {NET_KINDS}")
    # `spec` lets us rebuild the same architecture when loading a saved state_dict.
    net.spec = dict(kind=kind, clin_dim=int(clin_dim), backbone=backbone, embed_dim=int(embed_dim), pretrained=False)
    return net


def fit_late_fusion_weight(y_val, p_clin_val, p_img_val, grid=None) -> float:
    """Weight w in [0,1] for  p = w*p_clin + (1-w)*p_img, chosen by validation AUC (one parameter, low overfit risk)."""
    grid = np.linspace(0, 1, 21) if grid is None else grid
    aucs = [roc_auc_score(y_val, w * np.asarray(p_clin_val) + (1 - w) * np.asarray(p_img_val)) for w in grid]
    return float(grid[int(np.argmax(aucs))])
