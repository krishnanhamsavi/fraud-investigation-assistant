"""Phase 4 monitoring runner — drift detection + fairness analysis.

Usage:
    uv run python scripts/phase4_monitoring.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.data.load import PROCESSED_DIR
from src.data.preprocess import build_features
from src.models.champion import load_champion
from src.models.train import FEATURE_EXCLUDE
from src.monitoring.drift import simulate_deployment_drift, psi_label, PSI_MODERATE
from src.monitoring.fairness import run_fairness_suite

RESULTS_DIR = Path("docs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Proxy columns for fairness analysis (not protected classes — see docs/03_fairness_analysis.md)
PROXY_COLS = ["ProductCD", "card6", "DeviceType"]

print("=" * 62)
print("PHASE 4 — DRIFT & FAIRNESS MONITORING")
print("=" * 62)

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
print("\nLoading data …")
train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

X_train = train[feature_cols]
y_train = train["isFraud"].astype(int)
X_val = val[feature_cols]
y_val = val["isFraud"].astype(int)

# Raw joined data for fairness (needs original string-valued proxy columns)
df_joined = pd.read_parquet(PROCESSED_DIR / "joined.parquet")
df_joined = build_features(df_joined)
boundary_dt = train["TransactionDT"].max()
val_raw = df_joined[df_joined["TransactionDT"] > boundary_dt].copy()

model = load_champion()

# -----------------------------------------------------------------------
# Drift monitoring
# -----------------------------------------------------------------------
print("\n--- SCORE & FEATURE DRIFT ---")
score_reports, feat_report = simulate_deployment_drift(
    X_train, X_val, model, feature_cols,
    top_n_features=30,
    output_dir=RESULTS_DIR,
)

# Print score drift summary
for label, report in score_reports.items():
    print(f"\n  [{label}]")
    print(f"    Score PSI          : {report['psi']:.5f}  → {report['psi_label']}")
    print(f"    KS statistic       : {report['ks_statistic']:.5f}  (p={report['p_value']:.4f})")
    print(f"    Drift detected     : {report['drift_detected']}")
    print(f"    Mean score shift   : {report['ref_mean_score']:.4f} → {report['cur_mean_score']:.4f}")
    print(f"    Action required    : {report['action_required']}")

# Print top-10 features by PSI
print(f"\n  Top 10 features by PSI (train → val):")
print(f"  {'Feature':<22} {'PSI':>8}  {'Stability':<20}  {'Mean shift %':>12}")
print("  " + "-" * 68)
for _, row in feat_report.head(10).iterrows():
    print(
        f"  {row['feature']:<22} {row['psi']:>8.5f}  {row['stability']:<20}"
        f"  {row['mean_shift_pct']:>12.1f}%"
    )

n_sig = (feat_report["stability"] == "significant_drift").sum()
n_mod = (feat_report["stability"] == "moderate_drift").sum()
print(f"\n  Summary: {n_sig} features with significant drift, {n_mod} with moderate drift")

# Save score reports
(RESULTS_DIR / "score_drift.json").write_text(json.dumps(score_reports, indent=2))
print(f"  Saved → docs/results/score_drift.json")
print(f"  Saved → docs/results/feature_psi.csv")

# -----------------------------------------------------------------------
# Fairness analysis
# -----------------------------------------------------------------------
print("\n--- FAIRNESS ANALYSIS (PROXY VARIABLES) ---")

# Compute threshold at 3% review rate on val scores
y_score_val = model.predict_proba(X_val)[:, 1]
n_review = int(len(y_score_val) * 0.03)
threshold_3pct = float(np.sort(y_score_val)[::-1][n_review - 1])
print(f"\n  Threshold @ 3% review rate: {threshold_3pct:.4f}")

fairness_results = run_fairness_suite(
    y_val, y_score_val, val_raw, PROXY_COLS, threshold=threshold_3pct
)

for proxy, report in fairness_results.items():
    if not report:
        continue
    di = report["disparate_impact"]
    fpr_eq = report["fpr_equality"]
    print(f"\n  Proxy: {proxy}")
    print(f"    DI ratio (flag rates)  : {di['di_ratio']:.4f}"
          f"  ({'⚠ adverse impact indicated' if di['adverse_impact_indicated'] else '✓ within 4/5ths rule'})")
    print(f"    FPR gap                : {fpr_eq['fpr_gap_pct_pts']:.2f} ppt "
          f"({fpr_eq['min_fpr_group']} → {fpr_eq['max_fpr_group']})")
    print(f"\n    {'Group':<20} {'N':>7} {'FraudRate':>10} {'FlagRate':>10} {'FPR':>8} {'TPR':>8}")
    print("    " + "-" * 65)
    for _, row in report["group_metrics"].iterrows():
        print(
            f"    {str(row['group']):<20} {row['n_total']:>7,} {row['fraud_rate']:>10.2%}"
            f" {row['flag_rate']:>10.2%} {row['fpr']:>8.2%} {row['tpr']:>8.2%}"
        )

# Save fairness results (convert DataFrames to dicts)
fairness_json = {}
for proxy, report in fairness_results.items():
    if not report:
        continue
    fairness_json[proxy] = {
        "disparate_impact": report["disparate_impact"],
        "fpr_equality": report["fpr_equality"],
        "group_metrics": report["group_metrics"].to_dict(orient="records"),
    }
(RESULTS_DIR / "fairness_results.json").write_text(json.dumps(fairness_json, indent=2))
print(f"\n  Saved → docs/results/fairness_results.json")

print("\n" + "=" * 62)
print("Phase 4 monitoring complete.")
print("=" * 62)
