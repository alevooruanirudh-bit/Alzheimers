"""
Evaluate every trained model on the held-out TEST set:  python -m src.evaluate

This is the only place where test labels are used, and nothing is fitted
here: models, calibrators and thresholds were all fixed by src/train.py.

Outputs
  reports/results.json            all metrics (test + CV mean/std)
  reports/model_comparison.csv    one row per model
  reports/ablation.csv            contribution of each component
  reports/risk_levels.csv         test patients per risk level + actual AD rate
  reports/ceiling_analysis.json   label-noise ceiling (training data)
  reports/figures/*.png           confusion matrices, ROC, PR, calibration, loss curves
"""
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from statsmodels.stats.contingency_tables import mcnemar

from src import ceiling, config
from src.data import get_test
from src.metrics import binary_metrics
from src.models.bundle import ModelBundle

PROPOSED = "Hybrid Transformer + XGBoost"
COLORS = {
    "Logistic Regression": "#94a3b8", "SVM (RBF)": "#64748b", "Random Forest": "#0891b2",
    "XGBoost (plain)": "#0e7490", "XGBoost (tuned)": "#2563eb", "FT-Transformer": "#7c3aed",
    "FT-Transformer (no FE)": "#a78bfa", "Hybrid without feature engineering": "#14b8a6",
    PROPOSED: "#be123c",
}
ABLATION = [  # (row label, model name, what it shows)
    ("Transformer alone", "FT-Transformer", "FT-Transformer on engineered features"),
    ("XGBoost alone", "XGBoost (tuned)", "Tuned XGBoost on engineered features, no embeddings"),
    ("Hybrid without feature engineering", "Hybrid without feature engineering",
     "Transformer + XGBoost on raw features only"),
    ("Full hybrid (proposed)", PROPOSED, "Engineered features + Transformer embeddings + tuned XGBoost"),
]


def bootstrap_ci(y, pred, n: int = 2000, seed: int = config.SEED) -> list[float]:
    """95% bootstrap confidence interval of accuracy on the test set."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y) == np.asarray(pred)).astype(float)
    accs = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def plot_confusion(results: dict) -> None:
    names = list(results)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    for ax, name in zip(axes.ravel(), names):
        cm = np.array(results[name]["test"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No AD", "AD"], yticklabels=["No AD", "AD"])
        ax.set_title(f"{name}\nacc={results[name]['test']['accuracy']:.4f}", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrices_all.png", dpi=150)
    plt.close(fig)

    cm = np.array(results[PROPOSED]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 16},
                xticklabels=["No Alzheimer's", "Alzheimer's"], yticklabels=["No Alzheimer's", "Alzheimer's"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Proposed hybrid - test set")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix_hybrid.png", dpi=150)
    plt.close(fig)


def plot_curves(y, probs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS.get(name), lw=2.4 if name == PROPOSED else 1.4,
                label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - test set")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=COLORS.get(name), lw=2.4 if name == PROPOSED else 1.4, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - test set")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(y, raw: np.ndarray, calibrated: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, p, c in [("Hybrid (raw XGBoost output)", raw, "#94a3b8"),
                        ("Hybrid (calibrated)", calibrated, "#be123c")]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", color=c, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed Alzheimer's rate")
    ax.set_title("Calibration curve - test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)


def plot_loss_curves(summary: dict) -> None:
    tx = summary["models"]["FT-Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = tx["history_full"]
    axes[0].plot(h["train_loss"], label="training loss", color="#2563eb")
    axes[0].plot(h["val_loss"], label="validation loss", color="#0d9488")
    axes[0].axvline(tx["best_epoch_full"] - 1, ls="--", color="#be123c", label="best epoch (restored)")
    axes[0].set_title("Final FT-Transformer")
    for i, fh in enumerate(tx["fold_histories"]):
        axes[1].plot(fh["val_loss"], label=f"fold {i + 1}")
    axes[1].set_title("Validation loss of the 5 out-of-fold Transformers")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "transformer_loss_curves.png", dpi=150)
    plt.close(fig)


def risk_table(y, prob) -> pd.DataFrame:
    levels = pd.Series([config.risk_level(p) for p in prob], name="risk_level")
    df = pd.DataFrame({"risk_level": levels, "y": np.asarray(y)})
    out = df.groupby("risk_level").agg(patients=("y", "size"), alzheimers_cases=("y", "sum"))
    out["actual_alzheimers_rate"] = (out["alzheimers_cases"] / out["patients"]).round(4)
    out["share_of_test_set"] = (out["patients"] / len(df)).round(4)
    return out.reindex(["Low", "Medium", "High"]).fillna(0).reset_index()


def main() -> None:
    summary = json.loads((config.REPORTS_DIR / "training_summary.json").read_text())
    X_test, y_test = get_test()
    y = y_test.values
    print(f"Test patients: {len(y)}  (positive rate {y.mean():.3f})\n")

    results, probs, preds = {}, {}, {}
    for name, info in summary["models"].items():
        bundle = ModelBundle.load(info["file"])
        p = bundle.predict_proba(X_test)
        m = binary_metrics(y, p, bundle.threshold)
        pred = (p >= bundle.threshold).astype(int)
        m["accuracy_95ci"] = bootstrap_ci(y, pred)
        results[name] = {"group": info["group"], "test": m, "cv": info["cv"],
                         "calibration": info.get("calibration")}
        probs[name], preds[name] = p, pred
        if name == PROPOSED:
            hybrid_raw = bundle.predict_proba_raw(X_test)
        print(f"{name:36s} acc {m['accuracy']:.4f}  f1 {m['f1']:.4f}  auc {m['roc_auc']:.4f}  "
              f"(CV acc {info['cv']['accuracy']['mean']:.4f} +/- {info['cv']['accuracy']['std']:.4f})")

    # McNemar test: is the hybrid significantly different from tuned XGBoost?
    a, b = preds[PROPOSED] == y, preds["XGBoost (tuned)"] == y
    table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    mc = mcnemar(table, exact=True)

    # ---- tables ---------------------------------------------------------------
    rows = []
    for name, r in results.items():
        t, c = r["test"], r["cv"]
        rows.append({"model": name, "group": r["group"], "test_accuracy": t["accuracy"],
                     "test_accuracy_95ci": f"[{t['accuracy_95ci'][0]:.4f}, {t['accuracy_95ci'][1]:.4f}]",
                     "test_precision": t["precision"], "test_recall": t["recall"], "test_f1": t["f1"],
                     "test_roc_auc": t["roc_auc"], "threshold": t["threshold"],
                     "cv_accuracy_mean": c["accuracy"]["mean"], "cv_accuracy_std": c["accuracy"]["std"],
                     "cv_f1_mean": c["f1"]["mean"], "cv_f1_std": c["f1"]["std"],
                     "cv_roc_auc_mean": c["roc_auc"]["mean"], "cv_roc_auc_std": c["roc_auc"]["std"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False, float_format="%.4f")

    ablation = pd.DataFrame([{
        "configuration": label, "model": name, "description": desc,
        "test_accuracy": results[name]["test"]["accuracy"], "test_f1": results[name]["test"]["f1"],
        "test_roc_auc": results[name]["test"]["roc_auc"],
        "cv_accuracy_mean": results[name]["cv"]["accuracy"]["mean"],
        "cv_accuracy_std": results[name]["cv"]["accuracy"]["std"]} for label, name, desc in ABLATION])
    ablation.to_csv(config.REPORTS_DIR / "ablation.csv", index=False, float_format="%.4f")

    risk = risk_table(y, probs[PROPOSED])
    risk.to_csv(config.REPORTS_DIR / "risk_levels.csv", index=False)

    ceil = ceiling.analyse()
    rule_pred = ceiling.clinical_rule(X_test)
    ceil["rule_accuracy_test"] = float((rule_pred == y).mean())
    (config.REPORTS_DIR / "ceiling_analysis.json").write_text(json.dumps(ceil, indent=2))

    hybrid_acc = results[PROPOSED]["test"]["accuracy"]
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_test": int(len(y)), "test_positive_rate": float(y.mean()),
        "proposed_model": PROPOSED, "model_version": config.MODEL_VERSION,
        "target_accuracy": 0.981, "target_met": bool(hybrid_acc >= 0.981),
        "models": results,
        "ablation": ablation.to_dict(orient="records"),
        "risk_levels": risk.to_dict(orient="records"),
        "risk_thresholds": {"low_below": config.RISK_LOW_MAX, "high_above": config.RISK_HIGH_MIN},
        "mcnemar_hybrid_vs_xgb_tuned": {"table": table, "p_value": float(mc.pvalue)},
        "feature_selection": summary["feature_selection"]["chosen"],
        "ceiling_analysis": ceil,
    }
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(out, indent=2))

    # ---- figures ----------------------------------------------------------------
    plot_confusion(results)
    plot_curves(y, probs)
    plot_calibration(y, hybrid_raw, probs[PROPOSED])
    plot_loss_curves(summary)

    print("\nAblation:\n", ablation[["configuration", "test_accuracy", "cv_accuracy_mean"]].round(4).to_string(index=False))
    print("\nRisk levels (proposed hybrid, test set):\n", risk.to_string(index=False))
    print(f"\nMcNemar hybrid vs tuned XGBoost: p = {mc.pvalue:.3f}")
    print(f"Label-noise ceiling (train): rule acc {ceil['rule_accuracy_train']:.4f}, "
          f"exception AUC {ceil['exception_predictability_cv_auc']:.3f}; rule acc on test {ceil['rule_accuracy_test']:.4f}")
    print(f"\nProposed hybrid test accuracy {hybrid_acc:.4f} -> target 98.1% met: {out['target_met']}")

    # home-screening ("pre-test") scenarios: missing measurements, estimated scores
    from src import pretest
    print()
    pretest.evaluate()

    # explanations use the saved models and test set (nothing is fitted on test labels)
    from src import explain
    explain.main()


if __name__ == "__main__":
    main()
