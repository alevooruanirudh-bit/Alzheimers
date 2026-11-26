"""
Exploratory data analysis.

Prints shape, class balance and missing values, and saves a few plots to
reports/figures/. Run with `python -m src.eda`.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data import load_raw

sns.set_theme(style="whitegrid", palette="crest")


def main() -> None:
    df = load_raw()
    print("Shape:", df.shape)
    counts = df[config.TARGET].value_counts().sort_index()
    print("\nClass balance (0 = no Alzheimer's, 1 = Alzheimer's):")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v / len(df):.1%})")
    missing = df.isna().sum()
    print("\nMissing values:", int(missing.sum()), "in total")
    if missing.sum():
        print(missing[missing > 0])
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nNumerical summary:")
    print(df[config.NUMERICAL_FEATURES].describe().T[["mean", "std", "min", "max"]].round(2))

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=df[config.TARGET].map({0: "No Alzheimer's", 1: "Alzheimer's"}),
                  hue=df[config.TARGET], palette=["#0f766e", "#1d4ed8"], legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Correlation heatmap
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.7},
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print("\nCorrelation with Diagnosis (top 10 by |r|):")
    print(corr[config.TARGET].drop(config.TARGET).abs().sort_values(ascending=False).head(10).round(3))

    # 3. Distributions of key features split by diagnosis
    key = ["MMSE", "FunctionalAssessment", "ADL", "Age", "BMI", "SleepQuality"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), key):
        sns.kdeplot(data=df, x=col, hue=config.TARGET, fill=True, common_norm=False,
                    palette=["#0f766e", "#1d4ed8"], ax=ax)
        ax.set_title(col)
    fig.suptitle("Key feature distributions by diagnosis (1 = Alzheimer's)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_key_distributions.png", dpi=150)
    plt.close(fig)

    # 4. Symptom prevalence by diagnosis
    rates = df.groupby(config.TARGET)[config.SYMPTOM_FEATURES].mean().T
    rates.columns = ["No Alzheimer's", "Alzheimer's"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates.plot.barh(ax=ax, color=["#0f766e", "#1d4ed8"])
    ax.set_xlabel("Proportion of patients with symptom")
    ax.set_title("Symptom prevalence by diagnosis")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_symptom_prevalence.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
