"""Disparate impact analysis on proxy features.

Regulatory context
------------------
ECOA (Equal Credit Opportunity Act) and the Fair Housing Act prohibit credit
decisions that have an unjustified disparate impact on protected classes
(race, color, religion, national origin, sex, marital status, age).

The IEEE-CIS dataset contains no demographic information. Analysis here uses
PROXY variables (ProductCD, card6, DeviceType) to demonstrate the methodology
a bank would apply to real protected-class data. These proxies are NOT protected
classes; this analysis is a *process demonstration*, not a compliance assessment.

The 4/5ths rule (80% rule): if the flag rate for any group is less than 80% of
the group with the lowest flag rate, adverse impact is indicated and requires
investigation. For fraud models, "flagging" is the adverse action.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FOUR_FIFTHS_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def group_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    group: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Compute fairness metrics per group at a given score threshold.

    Metrics returned per group
    --------------------------
    n_total        : transaction count
    n_fraud        : actual fraud count
    fraud_rate     : actual fraud rate (base rate)
    flag_rate      : P(flagged) — the "adverse action" rate
    fpr            : false positive rate = P(flagged | legitimate)
    tpr            : true positive rate  = P(flagged | fraud) = recall
    precision      : P(fraud | flagged)
    di_numerator   : flag_rate used as numerator in disparate impact ratio
    """
    y_pred = (y_score >= threshold).astype(int)
    df = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
        "y_score": y_score,
        "group": group.values,
    })

    rows = []
    for grp, sub in df.groupby("group"):
        n = len(sub)
        n_fraud = sub["y_true"].sum()
        n_flagged = sub["y_pred"].sum()
        tp = ((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum()
        fp = ((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum()
        fn = ((sub["y_pred"] == 0) & (sub["y_true"] == 1)).sum()
        tn = ((sub["y_pred"] == 0) & (sub["y_true"] == 0)).sum()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        rows.append({
            "group": grp,
            "n_total": n,
            "n_fraud": int(n_fraud),
            "fraud_rate": round(float(n_fraud / n), 4),
            "flag_rate": round(float(n_flagged / n), 4),
            "fpr": round(float(fpr), 4),
            "tpr": round(float(tpr), 4),
            "precision": round(float(precision), 4),
        })

    result = pd.DataFrame(rows).sort_values("flag_rate", ascending=False).reset_index(drop=True)
    return result


def disparate_impact_ratio(metrics_df: pd.DataFrame) -> dict:
    """Compute the 4/5ths rule disparate impact ratio from group_metrics output.

    Returns the ratio of the LOWEST-flagged group to the HIGHEST-flagged group.
    A ratio < 0.80 indicates potential adverse impact under EEOC guidelines.
    """
    rates = metrics_df["flag_rate"].dropna()
    if len(rates) < 2:
        return {"ratio": np.nan, "adverse_impact_indicated": False}

    min_rate = rates.min()
    max_rate = rates.max()
    ratio = min_rate / max_rate if max_rate > 0 else np.nan

    return {
        "min_flag_group": metrics_df.loc[rates.idxmin(), "group"],
        "max_flag_group": metrics_df.loc[rates.idxmax(), "group"],
        "min_flag_rate": round(float(min_rate), 4),
        "max_flag_rate": round(float(max_rate), 4),
        "di_ratio": round(float(ratio), 4),
        "threshold_applied": FOUR_FIFTHS_THRESHOLD,
        "adverse_impact_indicated": bool(ratio < FOUR_FIFTHS_THRESHOLD),
    }


def fpr_equality_gap(metrics_df: pd.DataFrame) -> dict:
    """FPR equality gap — max FPR minus min FPR across groups.

    A large gap means legitimate transactions in one group are flagged at a
    much higher rate than another — an equity concern even if overall flag rates
    satisfy the 4/5ths rule (the 'equalised odds' fairness criterion).
    """
    fprs = metrics_df["fpr"].dropna()
    if len(fprs) < 2:
        return {"fpr_gap": np.nan}
    gap = float(fprs.max() - fprs.min())
    return {
        "max_fpr_group": metrics_df.loc[fprs.idxmax(), "group"],
        "min_fpr_group": metrics_df.loc[fprs.idxmin(), "group"],
        "max_fpr": round(float(fprs.max()), 4),
        "min_fpr": round(float(fprs.min()), 4),
        "fpr_gap": round(gap, 4),
        "fpr_gap_pct_pts": round(gap * 100, 2),
    }


# ---------------------------------------------------------------------------
# Full fairness report for a proxy variable
# ---------------------------------------------------------------------------

def fairness_report(
    y_true: pd.Series,
    y_score: np.ndarray,
    df_raw: pd.DataFrame,
    proxy_col: str,
    threshold: float,
    min_group_size: int = 100,
) -> dict:
    """End-to-end fairness report for one proxy variable.

    Groups with fewer than min_group_size transactions are merged into 'other'
    to ensure statistical reliability of per-group rates.

    Parameters
    ----------
    y_true      : true isFraud labels (val split)
    y_score     : model fraud probability scores
    df_raw      : raw validation DataFrame containing proxy_col (original strings)
    proxy_col   : column name to group by
    threshold   : score threshold (e.g. at 3% review rate)
    min_group_size : groups below this size are collapsed to 'other'
    """
    if proxy_col not in df_raw.columns:
        logger.warning("Proxy column %r not found — skipping fairness report.", proxy_col)
        return {}

    group = df_raw[proxy_col].copy()

    # Collapse small groups
    counts = group.value_counts()
    small = counts[counts < min_group_size].index
    group = group.replace(dict.fromkeys(small, "other"))
    group = group.fillna("missing")

    metrics = group_metrics(y_true, y_score, group, threshold)
    di = disparate_impact_ratio(metrics)
    fpr_gap = fpr_equality_gap(metrics)

    logger.info(
        "Fairness [%s] — DI ratio: %.3f  (adverse impact: %s)  FPR gap: %.3f",
        proxy_col, di["di_ratio"], di["adverse_impact_indicated"], fpr_gap["fpr_gap"],
    )

    return {
        "proxy_variable": proxy_col,
        "threshold": round(threshold, 4),
        "group_metrics": metrics,
        "disparate_impact": di,
        "fpr_equality": fpr_gap,
    }


# ---------------------------------------------------------------------------
# Multi-proxy summary
# ---------------------------------------------------------------------------

def run_fairness_suite(
    y_true: pd.Series,
    y_score: np.ndarray,
    df_raw: pd.DataFrame,
    proxy_cols: list[str],
    threshold: float,
) -> dict[str, dict]:
    """Run fairness_report for each proxy column; return keyed dict."""
    results = {}
    for col in proxy_cols:
        results[col] = fairness_report(y_true, y_score, df_raw, col, threshold)
    return results
