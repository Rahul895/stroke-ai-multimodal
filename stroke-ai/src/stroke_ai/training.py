"""Training / inference loops for the torch networks."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .utils import seed_everything


def _tensors(imgs, clin, idx=None, y=None):
    imgs = np.asarray(imgs, dtype=np.float32)
    if clin is None:
        clin = np.zeros((len(imgs), 1), dtype=np.float32)
    clin = np.asarray(clin, dtype=np.float32)
    if idx is not None:
        imgs, clin = imgs[idx], clin[idx]
    out = [torch.from_numpy(imgs), torch.from_numpy(clin)]
    if y is not None:
        out.append(torch.from_numpy(np.asarray(y, dtype=np.float32)[idx if idx is not None else slice(None)]))
    return out


@torch.no_grad()
def predict_net(model: nn.Module, imgs, clin=None, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    model.eval().to(device)
    xi, xc = _tensors(imgs, clin)
    out = []
    for s in range(0, len(xi), batch_size):
        out.append(torch.sigmoid(model(xi[s:s + batch_size].to(device), xc[s:s + batch_size].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def predict_net_mc(model: nn.Module, imgs, clin=None, T: int = 30, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """Monte-Carlo dropout: Dropout layers on, BatchNorm kept in eval. Returns (T, n) probabilities."""
    model.eval().to(device)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    xi, xc = _tensors(imgs, clin)
    samples = []
    for _ in range(T):
        out = []
        for s in range(0, len(xi), batch_size):
            out.append(torch.sigmoid(model(xi[s:s + batch_size].to(device), xc[s:s + batch_size].to(device))).cpu().numpy())
        samples.append(np.concatenate(out))
    model.eval()
    return np.stack(samples)


def train_net(model: nn.Module, imgs, clin, y, train_idx, val_idx, *, epochs: int = 25, lr: float = 1e-3,
              batch_size: int = 32, weight_decay: float = 1e-4, patience: int = 6, seed: int = 42,
              device: str = "cpu", verbose: bool = False) -> list[dict]:
    """Train with BCE (pos_weight for imbalance), early stopping on VALIDATION AUC; restores the best weights.
    `imgs` must already be normalised (per-scan standardisation, see data/imaging.py)."""
    seed_everything(seed)
    model.to(device)
    y = np.asarray(y).astype(int)
    n_pos = max(int((y[train_idx] == 1).sum()), 1)
    pos_weight = torch.tensor(float((y[train_idx] == 0).sum() / n_pos), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(*_tensors(imgs, clin, train_idx, y))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=len(ds) > batch_size)

    best_auc, best_state, bad, history = -1.0, None, 0, []
    vi = np.asarray(imgs)[val_idx]
    vc = None if clin is None else np.asarray(clin)[val_idx]
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xi, xc, yb in loader:
            xi, xc, yb = xi.to(device), xc.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xi, xc), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        val_p = predict_net(model, vi, vc, device)
        val_auc = roc_auc_score(y[val_idx], val_p) if len(np.unique(y[val_idx])) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / len(ds), "val_auc": float(val_auc)})
        if verbose:
            print(f"  epoch {epoch:02d} loss={total/len(ds):.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_state, bad = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
