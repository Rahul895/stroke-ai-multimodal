"""Explainability: SHAP for the clinical model, Grad-CAM for the image pathway."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from .clinical_model import feature_names, feature_matrix  # noqa: E402


# ------------------------------------------------------------------ SHAP (clinical)
class ClinicalExplainer:
    """SHAP values for a fitted clinical pipeline. Units: probability for random forest, log-odds for
    XGBoost / logistic regression. Positive = pushes risk up."""

    def __init__(self, pipe, background_df: pd.DataFrame):
        import shap

        self.pipe = pipe
        self.names = feature_names(pipe)
        self.clf = pipe.steps[-1][1]
        bg = feature_matrix(pipe, background_df)
        self._kind = "generic"
        if isinstance(self.clf, LogisticRegression):
            self.expl, self._kind = shap.LinearExplainer(self.clf, bg), "linear"
        else:
            try:
                self.expl, self._kind = shap.TreeExplainer(self.clf), "tree"
            except Exception:
                self.expl = shap.Explainer(lambda z: self.clf.predict_proba(z)[:, 1], bg)

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        z = feature_matrix(self.pipe, df)
        if self._kind == "generic":
            vals = np.asarray(self.expl(z).values)
        else:
            vals = self.expl.shap_values(z)
        if isinstance(vals, list):          # old shap: one array per class
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                  # new shap: (n, features, classes)
            vals = vals[:, :, -1]
        return pd.DataFrame(vals, columns=self.names)

    def top_features(self, df_row: pd.DataFrame, k: int = 5) -> list[dict]:
        sv = self.shap_values(df_row).iloc[0]
        z = pd.Series(feature_matrix(self.pipe, df_row)[0], index=self.names)
        order = sv.abs().sort_values(ascending=False).index[:k]
        return [{"feature": f, "contribution": float(sv[f]), "direction": "raises risk" if sv[f] > 0 else "lowers risk",
                 "scaled_value": float(z[f])} for f in order]


def plot_shap_bar(shap_df: pd.DataFrame, path: str, title: str = "Clinical features: mean |SHAP|") -> None:
    imp = shap_df.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ Grad-CAM (image)
def grad_cam(forward_fn, layer: torch.nn.Module, img: torch.Tensor) -> np.ndarray:
    """Grad-CAM heatmap in [0,1] with the input's spatial size.
    forward_fn(img_batch) -> logit tensor of shape (1,). img: (1, 1, H, W), already normalised."""
    acts, grads = [], []

    def fwd_hook(_m, _i, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = img.clone().requires_grad_(True)
            logit = forward_fn(x).sum()
            logit.backward()
    finally:
        handle.remove()
    a, g = acts[0].detach(), grads[0].detach()
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay_cam(img01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend heatmap onto a grayscale image. img01: (H, W) in [0,1]. Returns uint8 RGB (H, W, 3)."""
    rgb = np.stack([img01] * 3, axis=-1)
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (np.clip((1 - alpha) * rgb + alpha * heat, 0, 1) * 255).astype(np.uint8)


def png_bytes(arr: np.ndarray, scale: int = 4) -> bytes:
    im = Image.fromarray(arr)
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def plot_gradcam_grid(items: list[dict], path: str) -> None:
    """items: dicts with keys img (H,W), cam (H,W), title."""
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.8), squeeze=False)
    for j, it in enumerate(items):
        axes[0, j].imshow(it["img"], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(it["title"], fontsize=8)
        axes[1, j].imshow(overlay_cam(it["img"], it["cam"]))
        for r in (0, 1):
            axes[r, j].axis("off")
    axes[0, 0].text(-0.05, 0.5, "input", transform=axes[0, 0].transAxes, rotation=90, ha="right", va="center")
    axes[1, 0].text(-0.05, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
