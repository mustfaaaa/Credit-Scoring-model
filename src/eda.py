"""Exploratory Data Analysis (STEP 4).

Each figure answers one specific question that changes a modelling decision.
No decorative charts.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET
from src.data_loader import BILL_COLS, PAY_COLS

sns.set_theme(style="whitegrid")
COLORS = {0: "#2ca02c", 1: "#d62728"}  # green = repaid, red = defaulted


def _save(fig, filename):
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_target_distribution(df: pd.DataFrame, filename="eda_target_distribution.png"):
    """Q: How imbalanced is the target, and can we trust accuracy?"""
    counts = df[TARGET].value_counts().sort_index()
    share = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Repaid (0)", "Defaulted (1)"], counts.to_numpy(),
                  color=[COLORS[0], COLORS[1]])
    for bar, c, s in zip(bars, counts, share):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 300,
                format(int(c), ",") + "\n" + format(s * 100, ".2f") + "%",
                ha="center", fontsize=10)
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Target distribution: default payment next month\n"
                 "Imbalanced -- a majority-class model already scores "
                 + format(share.iloc[0] * 100, ".1f") + "% accuracy")
    return _save(fig, filename)


def plot_payment_history_vs_default(df: pd.DataFrame,
                                    filename="eda_payment_history.png"):
    """Q: Is repayment history the dominant risk signal? (Drives feature design.)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    rate = df.groupby("PAY_0")[TARGET].agg(["mean", "size"])
    rate = rate[rate["size"] >= 30]
    ax1.bar(rate.index.astype(str), rate["mean"].to_numpy(), color="#d62728")
    ax1.axhline(df[TARGET].mean(), ls="--", c="k",
                label="Overall default rate = "
                      + format(df[TARGET].mean(), ".3f"))
    ax1.set_xlabel("PAY_0  (September repayment status; >=1 means months late)")
    ax1.set_ylabel("Default rate")
    ax1.set_title("Default rate by most recent repayment status")
    ax1.legend(fontsize=9)

    n_late = (df[PAY_COLS] >= 1).sum(axis=1)
    rate2 = df.assign(n_late=n_late).groupby("n_late")[TARGET].mean()
    ax2.bar(rate2.index.astype(str), rate2.to_numpy(), color="#ff7f0e")
    ax2.axhline(df[TARGET].mean(), ls="--", c="k")
    ax2.set_xlabel("Number of late months out of 6")
    ax2.set_ylabel("Default rate")
    ax2.set_title("Default rate by count of delinquent months\n"
                  "(motivates the engineered n_delinquent_months feature)")
    return _save(fig, filename)


def plot_financial_profile(df: pd.DataFrame, filename="eda_financial_profile.png"):
    """Q: Do credit limit, age and utilisation separate the two classes?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for cls in (0, 1):
        sub = df[df[TARGET] == cls]
        axes[0].hist(sub["LIMIT_BAL"] / 1000, bins=40, alpha=0.55,
                     color=COLORS[cls], density=True,
                     label="Defaulted" if cls else "Repaid")
    axes[0].set_xlabel("Credit limit (thousand NT$)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Credit limit by outcome\nLower limits skew toward default")
    axes[0].legend()

    for cls in (0, 1):
        sub = df[df[TARGET] == cls]
        axes[1].hist(sub["AGE"], bins=35, alpha=0.55, color=COLORS[cls],
                     density=True, label="Defaulted" if cls else "Repaid")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Age by outcome\nWeak separation -- age alone is a poor signal")
    axes[1].legend()

    util = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-0.5, 2.0)
    for cls in (0, 1):
        axes[2].hist(util[df[TARGET] == cls], bins=40, alpha=0.55,
                     color=COLORS[cls], density=True,
                     label="Defaulted" if cls else "Repaid")
    axes[2].set_xlabel("Credit utilisation (BILL_AMT1 / LIMIT_BAL)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Utilisation by outcome\nDefaulters cluster near/above the limit")
    axes[2].legend()

    return _save(fig, filename)


def plot_categorical_rates(df: pd.DataFrame, filename="eda_categorical_rates.png"):
    """Q: Do the demographic categories carry usable signal?"""
    cats = ["EDUCATION", "MARRIAGE", "SEX"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    overall = df[TARGET].mean()

    for ax, col in zip(axes, cats):
        rate = df.groupby(col)[TARGET].mean().sort_values()
        ax.bar(rate.index.astype(str), rate.to_numpy(), color="#1f77b4")
        ax.axhline(overall, ls="--", c="k")
        ax.set_title(col + " vs default rate")
        ax.set_ylabel("Default rate")
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle("Default rate by demographic category "
                 "(dashed line = overall rate)", fontsize=12)
    return _save(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename="eda_correlation_heatmap.png"):
    """Q: Which features are redundant, and what correlates with the target?"""
    cols = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS[:3] + [
        "PAY_AMT1", "PAY_AMT2", TARGET
    ]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap\n"
                 "BILL_AMT columns are near-collinear; PAY_* correlate most "
                 "with the target", fontsize=11)
    return _save(fig, filename)


def run_all(df: pd.DataFrame) -> list[str]:
    """Generate every EDA figure and return the saved paths."""
    return [
        plot_target_distribution(df),
        plot_payment_history_vs_default(df),
        plot_financial_profile(df),
        plot_categorical_rates(df),
        plot_correlation_heatmap(df),
    ]


if __name__ == "__main__":
    from src.data_loader import load_clean

    for p in run_all(load_clean()):
        print("saved", p)
