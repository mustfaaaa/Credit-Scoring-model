"""Metrics, threshold selection and plotting (STEPS 14-17).

Nothing here is hard-coded: every number written to disk is produced by these
functions from real model predictions.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import COST_FN, COST_FP, FIGURES_DIR

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


# --------------------------------------------------------------- metrics ----
def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Standard classification metrics for the positive class (default = 1)."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    # A constant classifier produces no usable ranking, so AUC is undefined.
    if y_proba is not None and len(np.unique(y_proba)) > 1:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
        metrics["PR-AUC"] = average_precision_score(y_true, y_proba)
    else:
        metrics["ROC-AUC"] = float("nan")
        metrics["PR-AUC"] = float("nan")
    return metrics


def expected_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """Total business cost implied by a confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def best_threshold_by_cost(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Scan thresholds, return ``(threshold, cost)`` minimising total cost.

    Must be called with **out-of-fold training predictions only** -- choosing a
    threshold on the test set would amount to tuning on the test set.
    """
    grid = np.linspace(0.05, 0.95, 181)
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), cost_fn, cost_fp)
        for t in grid
    ]
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


# ----------------------------------------------------------------- plots ----
def _save(fig, filename: str) -> str:
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion_matrices(results: dict, filename="confusion_matrices.png"):
    """Grid of confusion matrices, one per model, annotated with raw counts."""
    names = list(results)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        cm = confusion_matrix(results[name]["y_true"], results[name]["y_pred"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j, i, format(int(v), ","), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11,
            )
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1], ["Pred: repay", "Pred: default"], fontsize=8)
        ax.set_yticks([0, 1], ["True: repay", "True: default"], fontsize=8)
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Confusion matrices on the held-out test set", fontsize=13)
    return _save(fig, filename)


def plot_roc_curves(results: dict, filename="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, r) in enumerate(results.items()):
        if r["y_proba"] is None or np.isnan(r["metrics"]["ROC-AUC"]):
            continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
        auc = r["metrics"]["ROC-AUC"]
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                label=name + " (AUC = " + format(auc, ".3f") + ")")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves -- test set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_pr_curves(results: dict, y_test, filename="precision_recall_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, r) in enumerate(results.items()):
        if r["y_proba"] is None or np.isnan(r["metrics"]["PR-AUC"]):
            continue
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_proba"])
        ap = r["metrics"]["PR-AUC"]
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)],
                label=name + " (AP = " + format(ap, ".3f") + ")")
    base = float(np.mean(y_test))
    ax.axhline(base, ls="--", c="k", lw=1,
               label="Default rate = " + format(base, ".3f"))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves -- test set\n"
                 "(more informative than ROC under class imbalance)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_model_comparison(table: pd.DataFrame, filename="model_comparison.png"):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(table))
    width = 0.16
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, table[m].to_numpy(), width,
               label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * 2, list(table.index), rotation=15,
                  ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(ncols=5, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


def plot_feature_importance(names, values, title, filename, top_n=20):
    order = np.argsort(values)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(order) + 1.8))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, filename)


def plot_threshold_analysis(y_true, y_proba, chosen, filename="threshold_analysis.png"):
    """Show how precision/recall/F1 and business cost trade off with threshold."""
    grid = np.linspace(0.05, 0.95, 91)
    prec, rec, f1s, costs = [], [], [], []
    for t in grid:
        pred = (y_proba >= t).astype(int)
        prec.append(precision_score(y_true, pred, zero_division=0))
        rec.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
        costs.append(expected_cost(y_true, pred))

    label = "Chosen = " + format(chosen, ".2f")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(grid, prec, label="Precision", color=PALETTE[0])
    ax1.plot(grid, rec, label="Recall", color=PALETTE[1])
    ax1.plot(grid, f1s, label="F1", color=PALETTE[2])
    ax1.axvline(chosen, ls="--", c="k", label=label)
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Metric trade-off vs decision threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(grid, costs, color=PALETTE[3])
    ax2.axvline(chosen, ls="--", c="k", label=label)
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Total cost (" + format(COST_FN, ".0f") + "xFN + "
                   + format(COST_FP, ".0f") + "xFP)")
    ax2.set_title("Business cost vs decision threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return _save(fig, filename)
