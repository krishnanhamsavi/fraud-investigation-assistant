"""Training entry point — trains champion, evaluates, saves metrics to JSON.

Usage (from project root):
    uv run python -m src.models.train
    uv run python -m src.models.train --data-dir data/processed --no-shap
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.champion import CHAMPION_PATH, load_champion, train_champion

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("docs/results")
FEATURE_EXCLUDE = {"TransactionID", "isFraud"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Load train/val parquet; return X_train, y_train, X_val, y_val, feature_cols."""
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")

    feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

    X_train = train[feature_cols]
    y_train = train["isFraud"].astype(int)
    X_val = val[feature_cols]
    y_val = val["isFraud"].astype(int)

    logger.info(
        "Loaded — train: %d×%d, val: %d×%d",
        X_train.shape[0], X_train.shape[1],
        X_val.shape[0], X_val.shape[1],
    )
    return X_train, y_train, X_val, y_val, feature_cols


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def fdr_at_review_rate(y_true: pd.Series, y_score: np.ndarray, rate: float) -> float:
    """Fraud Detection Rate: share of all fraud caught in top (rate*100)% scored txns.

    Operationally: if your team can review k% of daily volume, what fraction of
    fraud cases land in that review queue?
    """
    n_review = max(1, int(len(y_score) * rate))
    top_idx = np.argsort(y_score)[::-1][:n_review]
    return float(y_true.iloc[top_idx].sum() / y_true.sum())


def threshold_table(y_true: pd.Series, y_score: np.ndarray) -> list[dict]:
    """FDR, precision, recall at a range of review rates (operational threshold table)."""
    review_rates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    rows = []
    for rate in review_rates:
        n = max(1, int(len(y_score) * rate))
        top_idx = np.argsort(y_score)[::-1][:n]
        tp = int(y_true.iloc[top_idx].sum())
        total_fraud = int(y_true.sum())
        threshold = float(np.sort(y_score)[::-1][n - 1])
        rows.append({
            "review_rate_pct": round(rate * 100, 1),
            "threshold": round(threshold, 4),
            "n_reviewed": n,
            "fraud_caught": tp,
            "fdr": round(tp / total_fraud, 4),
            "precision": round(tp / n, 4),
        })
    return rows


def compute_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> dict:
    """Compute all headline metrics for the model card."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "fdr_at_3pct_review": round(fdr_at_review_rate(y_true, y_score, 0.03), 4),
        "fdr_at_5pct_review": round(fdr_at_review_rate(y_true, y_score, 0.05), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(
    data_dir: Path = Path("data/processed"),
    force_retrain: bool = False,
) -> dict:
    """Train champion (or load cached), evaluate, persist results. Return metrics dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, feature_cols = load_splits(data_dir)

    if CHAMPION_PATH.exists() and not force_retrain:
        logger.info("Existing champion found at %s — loading (use --force to retrain)", CHAMPION_PATH)
        model = load_champion()
    else:
        model = train_champion(X_train, y_train, X_val, y_val)

    logger.info("Scoring validation set …")
    y_score = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_score)
    tbl = threshold_table(y_val, y_score)

    result = {
        "model": "champion_xgboost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "n_features": len(feature_cols),
        "val_rows": len(y_val),
        "val_fraud_rate": round(float(y_val.mean()), 4),
        "val_metrics": metrics,
        "threshold_table": tbl,
    }

    out_path = RESULTS_DIR / "champion_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Metrics saved → %s", out_path)

    # Pretty-print summary
    print("\n" + "=" * 56)
    print("CHAMPION MODEL — VALIDATION METRICS")
    print("=" * 56)
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC           : {metrics['pr_auc']:.4f}")
    print(f"  FDR @ 3% review  : {metrics['fdr_at_3pct_review']:.2%}")
    print(f"  FDR @ 5% review  : {metrics['fdr_at_5pct_review']:.2%}")
    print(f"  Best iteration   : {result['best_iteration']}")
    print("\n  Review-rate threshold table:")
    print(f"  {'Rate':>6}  {'Threshold':>10}  {'FDR':>8}  {'Precision':>10}  {'N reviewed':>11}")
    for row in tbl:
        print(
            f"  {row['review_rate_pct']:>5.1f}%  {row['threshold']:>10.4f}"
            f"  {row['fdr']:>8.2%}  {row['precision']:>10.2%}"
            f"  {row['n_reviewed']:>11,}"
        )
    print("=" * 56)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train and evaluate champion model")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed parquet files")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    args = parser.parse_args()

    run_training(data_dir=Path(args.data_dir), force_retrain=args.force)
