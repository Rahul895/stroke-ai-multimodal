"""End-to-end experiment: data -> patient-level split -> clinical grid -> MRI net -> fusion variants ->
ablation table (with CIs) -> calibration -> external validation -> robustness -> subgroup analysis ->
uncertainty -> SHAP / Grad-CAM -> saved model for the API and demo.

Model / threshold selection uses the VALIDATION split only. The TEST split is touched once, at the end.

    python -m stroke_ai.pipeline --data synthetic --out artifacts
    python -m stroke_ai.pipeline --data csv --data-csv my_paired.csv --out artifacts
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .calibration import Calibrator  # noqa: E402
from .clinical_model import IMBALANCE, MODELS, feature_matrix, make_clinical_pipeline, predict_clinical  # noqa: E402
from .data.clinical import ensure_clinical_columns  # noqa: E402
from .data.imaging import normalize  # noqa: E402
from .data.real import load_paired_csv  # noqa: E402
from .data.splits import patient_level_split  # noqa: E402
from .data.synthetic import make_synthetic_cohort  # noqa: E402
from .explain import ClinicalExplainer, plot_gradcam_grid, plot_shap_bar  # noqa: E402
from .fairness import subgroup_table  # noqa: E402
from .fusion import build_net, fit_late_fusion_weight  # noqa: E402
from .inference import DISCLAIMER, Scorer, scorer_gradcam  # noqa: E402
from .metrics import bootstrap_ci, compute_metrics, paired_bootstrap_auc_diff, youden_threshold  # noqa: E402
from .robustness import robustness_table  # noqa: E402
from .training import train_net  # noqa: E402
from .utils import seed_everything  # noqa: E402

FUSION_NAMES = ["late_fusion", "feature_fusion", "feature_fusion_featsel", "cross_attention"]


@dataclass
class RunConfig:
    data: str = "synthetic"            # synthetic | csv
    data_csv: str | None = None
    image_root: str | None = None
    external_csv: str | None = None    # optional second site/dataset, same CSV format
    n_samples: int = 800
    img_size: int = 64
    backbone: str = "small_cnn"        # small_cnn | resnet18 | densenet121 | efficientnet_b0
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 6
    seed: int = 42
    out_dir: str = "artifacts"
    n_boot: int = 300
    mc_samples: int = 20
    use_mlflow: bool = False
    device: str = "cpu"


def load_data(cfg: RunConfig):
    if cfg.data == "synthetic":
        df, imgs = make_synthetic_cohort(cfg.n_samples, cfg.img_size, cfg.seed)
        ext = make_synthetic_cohort(max(200, cfg.n_samples // 2), cfg.img_size, cfg.seed + 1000, shift=True)
    elif cfg.data == "csv":
        if not cfg.data_csv:
            raise ValueError("--data csv needs --data-csv")
        df, imgs = load_paired_csv(cfg.data_csv, cfg.image_root, cfg.img_size)
        ext = load_paired_csv(cfg.external_csv, cfg.image_root, cfg.img_size) if cfg.external_csv else None
    else:
        raise ValueError("data must be 'synthetic' or 'csv'")
    return df.reset_index(drop=True), imgs, ext


def _row(name, y_val, pv, y_te, pt, n_boot, seed):
    thr = youden_threshold(y_val, pv)          # threshold from VALIDATION only
    m = compute_metrics(y_te, pt, thr)
    lo, hi = bootstrap_ci(y_te, pt, "roc_auc", n_boot, seed)
    plo, phi = bootstrap_ci(y_te, pt, "pr_auc", n_boot, seed)
    m.update(model=name, val_auc=float(roc_auc_score(y_val, pv)), auc_ci_low=lo, auc_ci_high=hi,
             pr_auc_ci_low=plo, pr_auc_ci_high=phi)
    return m


def run_pipeline(cfg: RunConfig, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    out = Path(cfg.out_dir)
    for sub in ("results", "figures", "model"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    # ------------------------------------------------------------ data + patient-level split
    df, imgs, ext = load_data(cfg)
    y = df["stroke"].to_numpy().astype(int)
    sp = patient_level_split(df, seed=cfg.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    log(f"rows: train={len(tr)} val={len(va)} test={len(te)} | prevalence={y.mean():.3f}")
    imgs_n = normalize(imgs)                                 # per-scan standardisation: no dataset statistics involved
    df_tr, df_va, df_te = df.iloc[tr], df.iloc[va], df.iloc[te]

    # ------------------------------------------------------------ clinical grid (model x imbalance strategy)
    grid, fitted = [], {}
    for m in MODELS:
        for imb in IMBALANCE:
            pipe = make_clinical_pipeline(m, imb, y_train=y[tr], seed=cfg.seed).fit(ensure_clinical_columns(df_tr), y[tr])
            pv = predict_clinical(pipe, df_va)
            fitted[(m, imb)] = pipe
            grid.append({"model": m, "imbalance": imb, "val_auc": roc_auc_score(y[va], pv)})
    grid_df = pd.DataFrame(grid).sort_values("val_auc", ascending=False)
    grid_df.to_csv(out / "results" / "clinical_grid_val.csv", index=False)
    best_m, best_imb = grid_df.iloc[0]["model"], grid_df.iloc[0]["imbalance"]
    log(f"best clinical (by val AUC): {best_m} + {best_imb}")
    clin_pipe = fitted[(best_m, best_imb)]
    sel_pipe = make_clinical_pipeline(best_m, best_imb, select_features=True, y_train=y[tr], seed=cfg.seed).fit(
        ensure_clinical_columns(df_tr), y[tr])

    # ------------------------------------------------------------ neural models
    def train_variant(kind, pipe=None, tag=""):
        xc = feature_matrix(pipe, df) if pipe is not None else None
        net = build_net(kind, xc.shape[1] if xc is not None else 1, cfg.backbone)
        log(f"training {kind}{tag} ...")
        train_net(net, imgs_n, xc, y, tr, va, epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                  patience=cfg.patience, seed=cfg.seed, device=cfg.device)
        return net

    img_net = train_variant("image_only")
    ff_net = train_variant("feature_fusion", clin_pipe)
    ffs_net = train_variant("feature_fusion", sel_pipe, " (selected features)")
    ca_net = train_variant("cross_attention", clin_pipe)

    kw = dict(device=cfg.device)
    scorers = {
        "clinical_only": Scorer("clinical", clin_pipe=clin_pipe, **kw),
        "clinical_only_featsel": Scorer("clinical", clin_pipe=sel_pipe, **kw),
        "mri_only": Scorer("mri", net=img_net, **kw),
    }
    w = fit_late_fusion_weight(y[va], scorers["clinical_only"].predict(df_va, imgs[va]), scorers["mri_only"].predict(df_va, imgs[va]))
    scorers["late_fusion"] = Scorer("late_fusion", clin_pipe=clin_pipe, net=img_net, late_w=w, **kw)
    scorers["feature_fusion"] = Scorer("feature_fusion", clin_pipe=clin_pipe, net=ff_net, **kw)
    scorers["feature_fusion_featsel"] = Scorer("feature_fusion", clin_pipe=sel_pipe, net=ffs_net, **kw)
    scorers["cross_attention"] = Scorer("cross_attention", clin_pipe=clin_pipe, net=ca_net, **kw)

    preds = {n: {"val": s.predict(df_va, imgs[va]), "test": s.predict(df_te, imgs[te])} for n, s in scorers.items()}

    # proposed model = best fusion variant by VALIDATION AUC (never by test)
    proposed = max(FUSION_NAMES, key=lambda n: roc_auc_score(y[va], preds[n]["val"]))
    log(f"proposed (best fusion on val): {proposed}")
    sc = scorers[proposed]

    # ------------------------------------------------------------ ablation on TEST
    abl = pd.DataFrame([_row(n, y[va], preds[n]["val"], y[te], preds[n]["test"], cfg.n_boot, cfg.seed) for n in preds])
    cols = ["model", "val_auc", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "sensitivity", "specificity",
            "f1", "balanced_accuracy", "mcc", "sens_at_90_spec", "brier", "ece", "accuracy", "threshold", "n", "n_pos"]
    abl[cols].to_csv(out / "results" / "ablation_test.csv", index=False)
    diffs = pd.DataFrame([
        {"comparison": f"{proposed} vs {other}",
         **paired_bootstrap_auc_diff(y[te], preds[proposed]["test"], preds[other]["test"], cfg.n_boot, cfg.seed)}
        for other in ("clinical_only", "mri_only", "late_fusion") if other != proposed])
    diffs.to_csv(out / "results" / "paired_auc_differences.csv", index=False)

    # ------------------------------------------------------------ calibration (fit on val, judge on test)
    cal = Calibrator("sigmoid").fit(y[va], preds[proposed]["val"])
    pv_cal, pt_cal = cal.transform(preds[proposed]["val"]), cal.transform(preds[proposed]["test"])
    thr = youden_threshold(y[va], pv_cal)
    final = _row(f"{proposed} (calibrated)", y[va], pv_cal, y[te], pt_cal, cfg.n_boot, cfg.seed)
    final["threshold"] = thr
    final.update(compute_metrics(y[te], pt_cal, thr))
    pd.DataFrame([{"stage": "raw", **{k: compute_metrics(y[te], preds[proposed]["test"], 0.5)[k] for k in ("brier", "ece")}},
                  {"stage": "calibrated (Platt)", "brier": final["brier"], "ece": final["ece"]}]
                 ).to_csv(out / "results" / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for label, pp in (("raw", preds[proposed]["test"]), ("calibrated", pt_cal)):
        fr, mp = calibration_curve(y[te], pp, n_bins=5, strategy="quantile")
        ax.plot(mp, fr, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend(); ax.set_title("Reliability (test)")
    fig.tight_layout(); fig.savefig(out / "figures" / "calibration.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------ external validation
    ext_table = None
    if ext is not None:
        ed, ei = ext
        pe = cal.transform(sc.predict(ed.reset_index(drop=True), ei))
        ye = ed["stroke"].to_numpy().astype(int)
        me = compute_metrics(ye, pe, thr)
        lo, hi = bootstrap_ci(ye, pe, "roc_auc", cfg.n_boot, cfg.seed)
        ext_table = pd.DataFrame([
            {"set": "internal test", "n": final["n"], "roc_auc": final["roc_auc"], "auc_ci_low": final["auc_ci_low"],
             "auc_ci_high": final["auc_ci_high"], "sensitivity": final["sensitivity"],
             "specificity": final["specificity"], "brier": final["brier"], "ece": final["ece"]},
            {"set": "external", "n": me["n"], "roc_auc": me["roc_auc"], "auc_ci_low": lo, "auc_ci_high": hi,
             "sensitivity": me["sensitivity"], "specificity": me["specificity"], "brier": me["brier"], "ece": me["ece"]}])
        ext_table.to_csv(out / "results" / "external_validation.csv", index=False)

    # ------------------------------------------------------------ robustness + subgroups
    rob = robustness_table({k: scorers[k] for k in ("clinical_only", "mri_only", proposed)}, df_te.reset_index(drop=True), imgs[te], y[te], cfg.seed)
    rob.to_csv(out / "results" / "robustness_test.csv", index=False)
    subgroup_table(df_te.reset_index(drop=True), y[te], pt_cal, thr).to_csv(out / "results" / "subgroups_test.csv", index=False)

    # ------------------------------------------------------------ uncertainty (MC dropout)
    def spread(df_, imgs_):
        s = sc.predict_samples(df_, imgs_, cfg.mc_samples)
        c = cal.transform(s.reshape(-1)).reshape(s.shape)
        return c.mean(0), c.std(0)
    _, sd_va = spread(df_va, imgs[va])
    mu_te, sd_te = spread(df_te, imgs[te])
    edges = [float(np.quantile(sd_va, 1 / 3)), float(np.quantile(sd_va, 2 / 3))]
    pd.DataFrame({"prob": mu_te, "std": sd_te, "y": y[te]}).to_csv(out / "results" / "uncertainty_test.csv", index=False)

    # ------------------------------------------------------------ explainability
    bg = ensure_clinical_columns(df_tr).sample(min(100, len(df_tr)), random_state=cfg.seed)
    expl = ClinicalExplainer(clin_pipe, bg)
    plot_shap_bar(expl.shap_values(df_te), str(out / "figures" / "shap_clinical.png"))
    items = []
    order = np.argsort(-pt_cal)
    for j in list(order[:3]) + list(order[-3:]):
        row = df_te.iloc[[j]]
        items.append({"img": imgs[te][j, 0], "cam": scorer_gradcam(sc, row, imgs[te][j]),
                      "title": f"y={y[te][j]}  p={pt_cal[j]:.2f}"})
    plot_gradcam_grid(items, str(out / "figures" / "gradcam.png"))
    fig, ax = plt.subplots(figsize=(5, 4))
    for k in ("clinical_only", "mri_only", proposed):
        from sklearn.metrics import roc_curve
        f, t, _ = roc_curve(y[te], preds[k]["test"]); ax.plot(f, t, label=f"{k} ({roc_auc_score(y[te], preds[k]['test']):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=8)
    ax.set_title("ROC (test)"); fig.tight_layout(); fig.savefig(out / "figures" / "roc.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------ persist deployable model
    mdir = out / "model"
    sc.save(mdir)
    joblib.dump(cal, mdir / "calibrator.joblib")
    joblib.dump(clin_pipe, mdir / "clinical_explainer.joblib")
    bg.to_csv(mdir / "shap_background.csv", index=False)
    decision = {"threshold": thr, "uncertainty_edges": edges, "img_size": cfg.img_size, "proposed": proposed,
                "late_weight": w, "backbone": cfg.backbone, "disclaimer": DISCLAIMER, "data": cfg.data}
    (mdir / "decision.json").write_text(json.dumps(decision, indent=2))
    summary = {"config": asdict(cfg), "proposed": proposed, "best_clinical": [str(best_m), str(best_imb)],
               "final_test": {k: final[k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "brier", "ece", "threshold", "n", "n_pos")},
               "final_auc_ci": [final["auc_ci_low"], final["auc_ci_high"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if cfg.use_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name=f"{cfg.data}-{proposed}"):
                mlflow.log_params({k: v for k, v in asdict(cfg).items() if v is not None})
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in summary["final_test"].items()})
                mlflow.log_artifacts(str(out / "results"), "results")
                mlflow.log_artifacts(str(out / "figures"), "figures")
        except ImportError:
            log("mlflow not installed: skipping tracking (pip install mlflow)")

    log("\nTEST ablation (threshold from validation):")
    log(abl[["model", "val_auc", "roc_auc", "auc_ci_low", "auc_ci_high", "sensitivity", "specificity"]].round(3).to_string(index=False))
    log(f"\nproposed={proposed}  calibrated test AUC={final['roc_auc']:.3f}  saved -> {out}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = RunConfig()
    for f, v in asdict(d).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, action="store_true")
        else:
            p.add_argument(flag, type=type(v) if v is not None else str, default=v)
    run_pipeline(RunConfig(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
