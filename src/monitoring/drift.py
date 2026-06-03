"""Population Stability Index (PSI) and KS-test drift detection.

PSI thresholds (industry standard):
  PSI < 0.10  → stable          — no action required
  PSI 0.10–0.20 → moderate drift — investigate, monitor closely
  PSI > 0.20  → significant drift — trigger retraining review

The PSI measures how much the distribution of a variable has shifted between a
reference period (model development) and a monitoring period (deployment window).
It is the primary signal used by banks to trigger model reviews under SR 11-7.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_STABLE = 0.10
PSI_MODERATE = 0.20


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def psi(
    reference: np.ndarray,
    current: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index between reference and current distributions.

    Uses reference-distribution quantiles as bin edges so that each reference
    bucket is equally populated. Epsilon smoothing prevents log(0) on empty bins.

    Parameters
    ----------
    reference : array from the development / baseline period
    current   : array from the monitoring / deployment period
    buckets   : number of equal-frequency bins (based on reference)
    epsilon   : small constant added to proportions to avoid log(0)
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove NaN — missingness is tracked separately
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Equal-frequency bin edges from the reference distribution
    quantiles = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Deduplicate edges that collapse when values are discrete
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0  # insufficient variation to measure

    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_label(psi_value: float) -> str:
    """Return the standard stability classification for a PSI value."""
    if np.isnan(psi_value):
        return "unknown"
    if psi_value < PSI_STABLE:
        return "stable"
    if psi_value < PSI_MODERATE:
        return "moderate_drift"
    return "significant_drift"


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test for score distribution drift.

    Returns
    -------
    dict with keys: ks_statistic, p_value, drift_detected
    Drift is flagged when KS > 0.05 AND p-value < 0.05.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    stat, pvalue = stats.ks_2samp(ref, cur)
    return {
        "ks_statistic": round(float(stat), 5),
        "p_value": round(float(pvalue), 6),
        "drift_detected": bool(stat > 0.05 and pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# Feature-level PSI report
# ---------------------------------------------------------------------------

def feature_psi_report(
    X_ref: pd.DataFrame,
    X_cur: pd.DataFrame,
    feature_cols: list[str],
    buckets: int = 10,
) -> pd.DataFrame:
    """Compute PSI and descriptive stats for each feature.

    Parameters
    ----------
    X_ref        : reference period feature matrix
    X_cur        : current/monitoring period feature matrix
    feature_cols : columns to evaluate
    buckets      : PSI bin count

    Returns
    -------
    DataFrame sorted descending by PSI, with columns:
    feature, psi, stability, ref_mean, cur_mean, mean_shift_pct,
    ref_missing_pct, cur_missing_pct
    """
    rows = []
    for col in feature_cols:
        if col not in X_ref.columns or col not in X_cur.columns:
            continue

        ref_series = X_ref[col]
        cur_series = X_cur[col]

        ref_miss = ref_series.isna().mean()
        cur_miss = cur_series.isna().mean()

        ref_vals = ref_series.dropna().values.astype(float)
        cur_vals = cur_series.dropna().values.astype(float)

        psi_val = psi(ref_vals, cur_vals, buckets=buckets)
        ref_mean = float(np.mean(ref_vals)) if len(ref_vals) else np.nan
        cur_mean = float(np.mean(cur_vals)) if len(cur_vals) else np.nan
        denom = abs(ref_mean) + 1e-8
        shift_pct = (cur_mean - ref_mean) / denom * 100 if not np.isnan(ref_mean) else np.nan

        rows.append({
            "feature": col,
            "psi": round(psi_val, 5),
            "stability": psi_label(psi_val),
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "mean_shift_pct": round(shift_pct, 2),
            "ref_missing_pct": round(ref_miss * 100, 1),
            "cur_missing_pct": round(cur_miss * 100, 1),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Score distribution monitoring
# ---------------------------------------------------------------------------

def score_drift_report(
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    window_label: str = "current",
) -> dict:
    """Full drift report for the model score distribution.

    Runs both PSI and KS test on the fraud probability scores.
    This is the primary signal for triggering a model review.
    """
    psi_val = psi(ref_scores, cur_scores, buckets=10)
    ks_result = ks_test(ref_scores, cur_scores)

    report = {
        "window": window_label,
        "psi": round(psi_val, 5),
        "psi_label": psi_label(psi_val),
        "ks_statistic": ks_result["ks_statistic"],
        "p_value": ks_result["p_value"],
        "drift_detected": ks_result["drift_detected"],
        "ref_mean_score": round(float(np.mean(ref_scores)), 5),
        "cur_mean_score": round(float(np.mean(cur_scores)), 5),
        "ref_p95_score": round(float(np.percentile(ref_scores, 95)), 5),
        "cur_p95_score": round(float(np.percentile(cur_scores, 95)), 5),
        "action_required": psi_val >= PSI_MODERATE or ks_result["drift_detected"],
    }
    return report


# ---------------------------------------------------------------------------
# Simulated drift runner (using time-split of validation data)
# ---------------------------------------------------------------------------

def simulate_deployment_drift(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    model,
    feature_cols: list[str],
    top_n_features: int = 30,
    output_dir: Path = Path("docs/results"),
) -> tuple[dict, pd.DataFrame]:
    """Simulate monitoring by comparing train vs val distributions.

    Splits val into early/late halves to show within-deployment temporal drift.
    Scores both halves with the model and runs PSI + KS on score distributions.

    Returns
    -------
    score_reports : dict with 'train_vs_val', 'early_val_vs_late_val' entries
    feature_report : DataFrame of per-feature PSI (train vs val)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Score distributions
    logger.info("Scoring reference (train) and current (val) sets …")
    ref_scores = model.predict_proba(X_train[feature_cols])[:, 1]
    cur_scores = model.predict_proba(X_val[feature_cols])[:, 1]

    # Early / late val split (by row order — already sorted by TransactionDT)
    mid = len(X_val) // 2
    early_scores = model.predict_proba(X_val.iloc[:mid][feature_cols])[:, 1]
    late_scores = model.predict_proba(X_val.iloc[mid:][feature_cols])[:, 1]

    score_reports = {
        "train_vs_val": score_drift_report(ref_scores, cur_scores, "val (deployment)"),
        "early_val_vs_late_val": score_drift_report(early_scores, late_scores, "late val"),
    }

    # Feature-level PSI (top features by SHAP importance if available, else first N)
    monitor_features = feature_cols[:top_n_features]
    logger.info("Computing feature PSI for %d features …", len(monitor_features))
    feat_report = feature_psi_report(X_train, X_val, monitor_features)

    # Save
    feat_csv = output_dir / "feature_psi.csv"
    feat_report.to_csv(feat_csv, index=False)
    logger.info("Feature PSI saved → %s", feat_csv)

    return score_reports, feat_report
