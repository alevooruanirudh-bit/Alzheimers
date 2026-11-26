"""
Global explainability report:  python -m src.explain   (also run by src.evaluate)

Produces in reports/figures/:
  shap_summary_hybrid.png        beeswarm, embedding dimensions grouped into one row
  shap_bar_hybrid.png            mean |SHAP| per (grouped) feature
  shap_summary_plain_xgb.png     beeswarm of the plain XGBoost on the ORIGINAL features
  shap_bar_plain_xgb.png
  attention_cls.png              average [CLS] -> feature attention of the Transformer
  attention_matrix.png           average feature-to-feature attention heatmap
  attention_vs_shap.png          top features by attention vs by SHAP
  example_waterfall.png / example_force.png   one test patient
and reports/explainability.json with the numbers behind the plots.

Explanations are computed on test patients using the already-trained models;
nothing is fitted here.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src import config
from src.data import get_test
from src.features import label_for
from src.predictor import EMB_GROUP, TXP_GROUP, RiskPredictor


def short_label(f: str) -> str:
    """Compact label for plots (interaction terms keep their column names)."""
    return f.replace("__x__", " x ") if "__x__" in f else label_for(f)


def grouped_shap_matrix(Xh: pd.DataFrame, values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum the SHAP values of embedding columns into one 'Transformer embedding' column."""
    emb_cols = [c for c in Xh.columns if c.startswith("tx_emb_")]
    sv = pd.DataFrame(values, columns=Xh.columns)
    grouped = sv.drop(columns=emb_cols + ["tx_prob"])
    grouped[EMB_GROUP] = sv[emb_cols].sum(axis=1)
    grouped[TXP_GROUP] = sv["tx_prob"]
    data = Xh.drop(columns=emb_cols + ["tx_prob"]).copy()
    data[EMB_GROUP] = np.nan  # a group of embedding dimensions has no single value
    data[TXP_GROUP] = Xh["tx_prob"]
    return grouped, data


def save_shap_plots(values: pd.DataFrame, data: pd.DataFrame, base: float, tag: str, title: str) -> None:
    exp = shap.Explanation(values=values.values, base_values=np.full(len(values), base),
                           data=data.values, feature_names=[label_for(c) for c in values.columns])
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title(title)
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_summary_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    plt.title(f"{title} - mean |SHAP|")
    plt.gcf().set_size_inches(9, 6.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_bar_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    pred = RiskPredictor()
    X_test, y_test = get_test()

    # ---- SHAP on the hybrid -------------------------------------------------
    Xh = pred.hybrid.features(X_test)
    sv = pred.explainer(Xh)
    base = float(np.ravel(sv.base_values)[0])
    g_values, g_data = grouped_shap_matrix(Xh, sv.values)
    save_shap_plots(g_values, g_data, base, "hybrid",
                    "Hybrid model SHAP (Transformer embedding grouped)")
    hybrid_importance = g_values.abs().mean().sort_values(ascending=False)
    emb_share = float(hybrid_importance[[EMB_GROUP, TXP_GROUP]].sum() / hybrid_importance.sum())

    # ---- SHAP on the plain XGBoost (original features) -----------------------
    sv_plain = pred.plain_explainer(X_test[config.RAW_FEATURES])
    save_shap_plots(pd.DataFrame(sv_plain.values, columns=config.RAW_FEATURES), X_test[config.RAW_FEATURES],
                    float(np.ravel(sv_plain.base_values)[0]), "plain_xgb",
                    "Plain XGBoost SHAP (original features)")
    plain_importance = pd.Series(np.abs(sv_plain.values).mean(0), index=config.RAW_FEATURES).sort_values(ascending=False)

    # ---- Transformer attention -------------------------------------------------
    t = pred.hybrid.transformer
    Xf = pred.hybrid.fe.transform(X_test)
    att_full = t.attention(Xf).mean(axis=0)                      # (tokens, tokens)
    names = t.token_names
    cls_att = pd.Series(att_full[0, 1:], index=names[1:]).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    top = cls_att.head(20)[::-1]
    ax.barh([short_label(f) for f in top.index], top.values, color="#0d9488")
    ax.set_xscale("log")
    ax.set_xlabel("Average attention weight from [CLS] (test patients)")
    ax.set_title("Transformer attention: which features the [CLS] token looks at")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "attention_cls.png", dpi=140)
    plt.close(fig)

    order = [0] + [names.index(f) for f in cls_att.index[:20]]
    sub = att_full[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(sub, cmap="viridis")
    lbl = ["[CLS]"] + [short_label(f)[:34] for f in cls_att.index[:20]]
    ax.set_xticks(range(len(lbl)), lbl, rotation=75, fontsize=7.5)
    ax.set_yticks(range(len(lbl)), lbl, fontsize=7.5)
    ax.set_xlabel("attended-to token (key)")
    ax.set_ylabel("attending token (query)")
    ax.set_title("Average feature-to-feature attention (all layers and heads)")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "attention_matrix.png", dpi=140)
    plt.close(fig)

    # ---- attention vs SHAP on the same engineered features ---------------------
    shap_feat = hybrid_importance.drop([EMB_GROUP, TXP_GROUP])
    common = [f for f in cls_att.index if f in shap_feat.index]
    rho = spearmanr(cls_att[common], shap_feat[common]).statistic
    k = 10
    cmp = pd.DataFrame({
        "rank": range(1, k + 1),
        "top_by_attention": [label_for(f) for f in cls_att.index[:k]],
        "attention": cls_att.values[:k].round(4),
        "top_by_shap": [label_for(f) for f in shap_feat.index[:k]],
        "mean_abs_shap": shap_feat.values[:k].round(4),
    })
    cmp.to_csv(config.REPORTS_DIR / "attention_vs_shap.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5))
    for ax, series, color, title in [
            (axes[0], cls_att.head(k), "#0d9488", "Top features by Transformer [CLS] attention"),
            (axes[1], shap_feat.head(k), "#2563eb", "Top features by hybrid SHAP (mean |SHAP|, embedding excluded)")]:
        sr = series[::-1]
        ax.barh([short_label(f) for f in sr.index], sr.values, color=color)
        for y_pos, v in enumerate(sr.values):
            ax.text(v, y_pos, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Attention vs SHAP (Spearman rank correlation over all features = {rho:.2f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "attention_vs_shap.png", dpi=140)
    plt.close(fig)

    # ---- example local explanation (highest-risk-uncertain test patient) --------
    probs = pred.hybrid.predict_proba(X_test)
    idx = int(np.argmin(np.abs(probs - 0.6)))
    res = pred.predict(X_test.iloc[idx].to_dict())
    (config.FIGURES_DIR / "example_waterfall.png").write_bytes(pred.waterfall_png(res))
    (config.FIGURES_DIR / "example_force.png").write_bytes(pred.force_png(res))

    out = {
        "hybrid_mean_abs_shap": hybrid_importance.round(5).to_dict(),
        "transformer_share_of_hybrid_shap": emb_share,
        "plain_xgb_mean_abs_shap": plain_importance.round(5).to_dict(),
        "cls_attention": cls_att.round(5).to_dict(),
        "attention_vs_shap_spearman": float(rho),
        "attention_vs_shap_top10": cmp.to_dict(orient="records"),
        "example_patient_test_row": idx,
        "example_patient_top_factors": res["top_increasing"] + res["top_decreasing"],
    }
    (config.REPORTS_DIR / "explainability.json").write_text(json.dumps(out, indent=2))
    print(f"\nExplainability: Transformer share of hybrid |SHAP| = {emb_share:.1%}; "
          f"attention-vs-SHAP Spearman = {rho:.2f}")
    print("Top hybrid SHAP features:", ", ".join(hybrid_importance.index[:6]))
    print("Top attention features:  ", ", ".join(cls_att.index[:6]))


if __name__ == "__main__":
    main()
