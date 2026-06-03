"""Phase 3 comparison harness — trains challengers, prints comparison table.

Usage (from project root):
    uv run python scripts/phase3_compare.py           # train + compare
    uv run python scripts/phase3_compare.py --force   # retrain even if cached
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.data.load import PROCESSED_DIR
from src.data.preprocess import build_features, time_split
from src.models.champion import CHAMPION_PATH, load_champion
from src.models.challenger import (
    MODEL_DIR,
    load_challenger_a, load_challenger_b,
    predict_lr,
    train_challenger_a, train_challenger_b,
)
from src.models.train import FEATURE_EXCLUDE, compute_metrics

RESULTS_DIR = Path("docs/results")


def _infer_latency_ms(predict_fn, X: pd.DataFrame, n_rows: int = 1000, n_reps: int = 10) -> float:
    """Median ms to score n_rows transactions (10 repetitions)."""
    sample = X.head(n_rows)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def main(force: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load processed data (Champion + Challenger B use ordinal-encoded parquets)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading processed splits …")
    train_proc = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_proc = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    feature_cols = [c for c in train_proc.columns if c not in FEATURE_EXCLUDE]

    X_train = train_proc[feature_cols]
    y_train = train_proc["isFraud"].astype(int)
    X_val = val_proc[feature_cols]
    y_val = val_proc["isFraud"].astype(int)

    # -----------------------------------------------------------------------
    # 2. Load raw joined data (Challenger A needs original strings for TargetEncoder)
    #    Use the same temporal boundary as the pre-processed split.
    # -----------------------------------------------------------------------
    print("Loading raw joined data for Challenger A …")
    df_raw = pd.read_parquet(PROCESSED_DIR / "joined.parquet")
    df_raw = build_features(df_raw)

    boundary_dt = train_proc["TransactionDT"].max()
    train_raw = df_raw[df_raw["TransactionDT"] <= boundary_dt].sort_values("TransactionDT")
    val_raw = df_raw[df_raw["TransactionDT"] > boundary_dt].sort_values("TransactionDT")

    X_train_raw = train_raw.drop(columns=list(FEATURE_EXCLUDE), errors="ignore")
    X_val_raw = val_raw.drop(columns=list(FEATURE_EXCLUDE), errors="ignore")
    y_train_raw = train_raw["isFraud"].astype(int)
    y_val_raw = val_raw["isFraud"].astype(int)

    print(f"  Raw split check — train: {len(train_raw):,}, val: {len(val_raw):,}")

    # -----------------------------------------------------------------------
    # 3. Champion (already trained)
    # -----------------------------------------------------------------------
    print("\n[1/3] Champion XGBoost …")
    champion = load_champion()
    t0 = time.perf_counter()
    y_score_champ = champion.predict_proba(X_val)[:, 1]
    print(f"  Scored {len(X_val):,} rows in {(time.perf_counter()-t0)*1000:.0f}ms")
    champ_latency = _infer_latency_ms(lambda X: champion.predict_proba(X)[:, 1], X_val)

    champ_saved = json.loads((RESULTS_DIR / "champion_metrics.json").read_text())
    champ_best_iter = champ_saved.get("best_iteration", "—")

    # -----------------------------------------------------------------------
    # 4. Challenger A — Logistic Regression
    # -----------------------------------------------------------------------
    print("\n[2/3] Challenger A — Logistic Regression …")
    lr_path = MODEL_DIR / "challenger_a_lr.joblib"
    if lr_path.exists() and not force:
        print("  Loading cached model …")
        lr_pipe, lr_features = load_challenger_a(lr_path)
        lr_train_time = "cached"
    else:
        t0 = time.perf_counter()
        lr_pipe, lr_features = train_challenger_a(X_train_raw, y_train_raw, lr_path)
        lr_train_time = f"{time.perf_counter() - t0:.0f}s"

    t0 = time.perf_counter()
    y_score_lr = predict_lr(lr_pipe, X_val_raw, lr_features)
    print(f"  Scored {len(X_val_raw):,} rows in {(time.perf_counter()-t0)*1000:.0f}ms")
    lr_latency = _infer_latency_ms(lambda X: predict_lr(lr_pipe, X, lr_features), X_val_raw)

    # -----------------------------------------------------------------------
    # 5. Challenger B — LightGBM
    # -----------------------------------------------------------------------
    print("\n[3/3] Challenger B — LightGBM …")
    lgbm_path = MODEL_DIR / "challenger_b_lgbm.txt"
    if lgbm_path.exists() and not force:
        print("  Loading cached model …")
        lgbm_booster, lgbm_features = load_challenger_b(lgbm_path)
        lgbm_train_time = "cached"
        y_score_lgbm = lgbm_booster.predict(X_val[lgbm_features].values)
    else:
        t0 = time.perf_counter()
        lgbm_model, lgbm_features = train_challenger_b(X_train, y_train, X_val, y_val, lgbm_path)
        lgbm_train_time = f"{time.perf_counter() - t0:.0f}s"
        y_score_lgbm = lgbm_model.predict_proba(X_val[lgbm_features])[:, 1]

    lgbm_booster_loaded, _ = load_challenger_b(lgbm_path)
    lgbm_latency = _infer_latency_ms(
        lambda X: lgbm_booster_loaded.predict(X[lgbm_features].values), X_val
    )

    # -----------------------------------------------------------------------
    # 6. Metrics for all three
    # -----------------------------------------------------------------------
    print("\nComputing metrics …")
    models = {
        "Champion (XGBoost)": (y_score_champ, y_val),
        "Challenger A (LR)": (y_score_lr, y_val_raw),
        "Challenger B (LightGBM)": (y_score_lgbm, y_val),
    }
    all_metrics = {name: compute_metrics(y_true, y_score)
                   for name, (y_score, y_true) in models.items()}

    latencies = {
        "Champion (XGBoost)": champ_latency,
        "Challenger A (LR)": lr_latency,
        "Challenger B (LightGBM)": lgbm_latency,
    }
    features_used = {
        "Champion (XGBoost)": len(feature_cols),
        "Challenger A (LR)": len(lr_features),
        "Challenger B (LightGBM)": len(lgbm_features),
    }

    # -----------------------------------------------------------------------
    # 7. Print comparison table
    # -----------------------------------------------------------------------
    W = 78
    print("\n" + "=" * W)
    print("MODEL COMPARISON — VALIDATION SET")
    print("=" * W)
    print(f"  {'Model':<26} {'ROC-AUC':>8} {'PR-AUC':>8} {'FDR@3%':>8} {'FDR@5%':>8} {'Lat(ms)':>9} {'Feats':>6}")
    print("  " + "-" * (W - 2))
    for name, m in all_metrics.items():
        print(
            f"  {name:<26} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}"
            f" {m['fdr_at_3pct_review']:>8.2%} {m['fdr_at_5pct_review']:>8.2%}"
            f" {latencies[name]:>9.1f} {features_used[name]:>6}"
        )
    print("=" * W)
    print("  Lat = median ms to score 1,000 transactions (10 reps)")

    # -----------------------------------------------------------------------
    # 8. Save JSON results
    # -----------------------------------------------------------------------
    result = {
        name: {**m, "latency_ms_per_1k": latencies[name], "n_features": features_used[name]}
        for name, m in all_metrics.items()
    }
    out_path = RESULTS_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")

    # -----------------------------------------------------------------------
    # 9. Inject metrics table into docs/02_challenger_analysis.md
    # -----------------------------------------------------------------------
    _update_challenger_doc(all_metrics, latencies, features_used)


def _update_challenger_doc(metrics: dict, latencies: dict, features: dict) -> None:
    """Write the metrics table into the challenger analysis document."""
    doc_path = Path("docs/02_challenger_analysis.md")
    if not doc_path.exists():
        return

    text = doc_path.read_text()
    header = (
        "| Model | ROC-AUC | PR-AUC | FDR@3% | FDR@5% | Latency (ms/1k) | Features |\n"
        "|-------|---------|--------|--------|--------|-----------------|----------|\n"
    )
    rows = ""
    for name, m in metrics.items():
        rows += (
            f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
            f" {m['fdr_at_3pct_review']:.2%} | {m['fdr_at_5pct_review']:.2%} |"
            f" {latencies[name]:.1f} | {features[name]} |\n"
        )
    table = header + rows

    marker_start = "<!-- METRICS_TABLE_START -->"
    marker_end = "<!-- METRICS_TABLE_END -->"
    if marker_start in text and marker_end in text:
        before = text[: text.index(marker_start) + len(marker_start)]
        after = text[text.index(marker_end):]
        text = before + "\n" + table + after
        doc_path.write_text(text)
        print(f"Updated metrics table in {doc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if cached models exist")
    args = parser.parse_args()
    main(force=args.force)
