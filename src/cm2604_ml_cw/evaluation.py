"""Cross-validation setup, metrics, result tables and plots for the churn models.

Used by 03_modelling.ipynb. Churn (1) is the positive class for every metric.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Project root is one level above src/, so figures always go to <repo>/figures
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Metrics recorded during cross-validation (churn = positive class).
# zero_division=0 avoids warnings for the baseline, which never predicts churn.
SCORING = {
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_cv(random_state, n_splits=5):
    """Stratified K-fold: every fold keeps the original ~26.5% churn rate."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_fig(name, enabled=True):
    """Save the current figure to figures/<name>.png (200 dpi) for the report."""
    if enabled:
        FIG_DIR.mkdir(exist_ok=True)
        plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def param_label(value):
    """Readable label for a hyperparameter value. None can come back from the search as NaN."""
    return "None" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def results_table(search, top=10):
    """Best hyperparameter combinations from a search, ranked by CV F1.

    train_cv_gap = training F1 - CV F1. A large gap means overfitting.
    The search must be run with scoring=SCORING, refit="f1" and return_train_score=True.
    """
    res = pd.DataFrame(search.cv_results_)
    params = [c for c in res.columns if c.startswith("param_")]
    table = res[params].rename(columns=lambda c: c.replace("param_", ""))
    table["train_f1"] = res["mean_train_f1"]
    table["cv_f1"] = res["mean_test_f1"]
    table["cv_f1_std"] = res["std_test_f1"]
    table["train_cv_gap"] = table["train_f1"] - table["cv_f1"]
    table["cv_recall"] = res["mean_test_recall"]
    table["cv_precision"] = res["mean_test_precision"]
    table["cv_roc_auc"] = res["mean_test_roc_auc"]
    table["rank"] = res["rank_test_f1"]
    return table.sort_values("rank").head(top).reset_index(drop=True).round(4)


def best_scores(search, mask=None):
    """Mean CV scores of the best combination.

    mask: optional function that takes the cv_results_ DataFrame and returns a
    boolean Series, e.g. lambda r: r["param_class_weight"].map(param_label) == "None"
    """
    res = pd.DataFrame(search.cv_results_)
    if mask is not None:
        res = res[mask(res)]
    best = res.loc[res["mean_test_f1"].idxmax()]
    return {m: best[f"mean_test_{m}"] for m in SCORING}


def plot_validation_curve(search, param, log_x=False, title=None):
    """Training vs CV F1 for one hyperparameter."""
    res = pd.DataFrame(search.cv_results_)
    res["_label"] = res[f"param_{param}"].map(param_label)
    best = res.sort_values("mean_test_f1", ascending=False).groupby("_label", sort=False).head(1)
    best = (best.assign(_num=pd.to_numeric(best["_label"], errors="coerce"))
                .sort_values(["_num", "_label"], na_position="last"))

    numeric = best["_num"].notna().all()
    x = best["_num"].to_numpy() if numeric else np.arange(len(best))
    cv_mean, cv_std = best["mean_test_f1"].to_numpy(), best["std_test_f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, best["mean_train_f1"], "o-", label="Training")
    ax.plot(x, cv_mean, "o-", label="Cross-validation")
    ax.fill_between(x, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2)
    if not numeric:
        ax.set_xticks(x, best["_label"])
    elif log_x:
        ax.set_xscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("F1")
    ax.set_title(title or f"Validation curve: {param}")
    ax.legend()
    plt.tight_layout()
    return fig


def evaluate(model, X, y, threshold=0.5):
    """Test-set metrics for one fitted model."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }