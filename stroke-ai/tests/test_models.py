import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from stroke_ai.data.synthetic import make_synthetic_cohort
from stroke_ai.explain import grad_cam, overlay_cam
from stroke_ai.fusion import build_net, fit_late_fusion_weight
from stroke_ai.robustness import degrade_images, mask_missing


@pytest.mark.parametrize("kind", ["image_only", "feature_fusion", "cross_attention"])
@pytest.mark.parametrize("backbone", ["small_cnn", "resnet18"])
def test_net_output_shape(kind, backbone):
    net = build_net(kind, 11, backbone).eval()
    out = net(torch.randn(4, 1, 64, 64), torch.randn(4, 11))
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_gradcam_range_and_shape():
    net = build_net("feature_fusion", 5).eval()
    clin = torch.randn(1, 5)
    cam = grad_cam(lambda x: net(x, clin), net.backbone.features, torch.randn(1, 1, 64, 64))
    assert cam.shape == (64, 64) and 0.0 <= cam.min() and cam.max() <= 1.0
    assert overlay_cam(np.random.rand(64, 64).astype(np.float32), cam).shape == (64, 64, 3)


def test_late_fusion_weight_is_valid():
    y = np.array([0, 0, 1, 1, 0, 1])
    pc, pi = np.array([.2, .3, .7, .8, .1, .9]), np.array([.5] * 6)
    w = fit_late_fusion_weight(y, pc, pi)
    assert 0.0 <= w <= 1.0
    assert roc_auc_score(y, w * pc + (1 - w) * pi) == 1.0           # never picks a weight that discards the informative model
    assert fit_late_fusion_weight(y, pi, pc) < 1.0


def test_missing_mask_fraction_and_degradations():
    df, imgs = make_synthetic_cohort(200, 32, seed=0)
    m = mask_missing(df.drop(columns=["bmi"]), 0.3, seed=1)
    assert 0.2 < m[["age", "hypertension", "gender"]].isna().mean().mean() < 0.4
    for kind, level in [("noise", 0.1), ("blur", 2.0), ("contrast", 0.5), ("lowres", 4)]:
        d = degrade_images(imgs, kind, level)
        assert d.shape == imgs.shape and d.min() >= 0 and d.max() <= 1
    assert np.allclose(degrade_images(imgs, "noise", 0.0), imgs) and np.allclose(degrade_images(imgs, "lowres", 1), imgs)
