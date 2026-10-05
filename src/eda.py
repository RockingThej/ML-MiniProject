"""Exploratory data analysis. Saves figures and a summary to results/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import CONTINUOUS, RESULTS, TARGET, load_data


def main():
    df = load_data()
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum().sum())
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))
    print("Sex counts (0=female,1=male):\n", df["sex"].value_counts())
    df.describe().T.to_csv(RESULTS / "eda_summary.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    plt.tight_layout(); plt.savefig(RESULTS / "eda_correlation.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, len(CONTINUOUS), figsize=(4 * len(CONTINUOUS), 7))
    for i, c in enumerate(CONTINUOUS):
        sns.histplot(data=df, x=c, hue=TARGET, kde=True, ax=axes[0, i])
        sns.boxplot(data=df, x=TARGET, y=c, ax=axes[1, i])
    plt.tight_layout(); plt.savefig(RESULTS / "eda_continuous.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, c in zip(axes, ["sex", "cp", "exng", "thall"]):
        sns.barplot(data=df, x=c, y=TARGET, ax=ax)
        ax.set_ylabel("Proportion high risk")
    plt.tight_layout(); plt.savefig(RESULTS / "eda_categorical.png", dpi=150); plt.close()
    print("EDA figures saved to", RESULTS)


if __name__ == "__main__":
    main()
