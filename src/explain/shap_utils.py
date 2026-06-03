"""SHAP explainability utilities — TreeExplainer + ECOA-style reason codes.

SR 11-7 / ECOA context
----------------------
Adverse action notices under ECOA (Reg B) require specific, actionable reason codes
when credit or service is denied. For a fraud score, the equivalent is providing
human-readable explanations of why a transaction was flagged. The reason code
mapping below translates SHAP feature contributions into that language.

Note: The V-features in the IEEE-CIS dataset are anonymised by Vesta Corporation.
The reason code assignments for V-features are based on their observed correlation
patterns and SHAP directionality, documented in docs/07_reason_codes.md.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ECOA-style adverse action reason codes
# ---------------------------------------------------------------------------

REASON_CODES: dict[str, str] = {
    "R01": "Unusual transaction amount relative to account history",
    "R02": "High transaction velocity or frequency on this card",
    "R03": "Email domain associated with elevated fraud risk",
    "R04": "Transaction inconsistent with typical geographic pattern",
    "R05": "Device characteristics inconsistent with account profile",
    "R06": "Transaction pattern deviates from established account behavior",
    "R07": "Time-of-day or day-of-week anomaly for this account",
    "R08": "Billing or shipping address information inconsistency",
    "R09": "Card type or issuer profile associated with elevated risk",
    "R10": "Multiple identity fields could not be verified",
    "R11": "Product category associated with elevated fraud risk",
    "R12": "Aggregate risk score across behavioral dimensions elevated",
    "R13": "Transaction amount exceeds typical category spend",
    "R14": "Cross-channel or cross-device activity pattern detected",
    "R15": "Account-level risk signal from historical transaction graph",
}

# Direct feature → reason code mapping for named features
_FEATURE_CODE_MAP: dict[str, str] = {
    "TransactionAmt":     "R01",
    "TransactionAmt_log": "R01",
    "ProductCD":          "R11",
    "card1":  "R09", "card2": "R09", "card3": "R09",
    "card4":  "R09", "card5": "R09", "card6": "R09",
    "addr1":  "R08", "addr2": "R08",
    "dist1":  "R04", "dist2": "R04",
    "P_emaildomain":    "R03",
    "R_emaildomain":    "R03",
    "uid_card1_email":  "R09",
    "tx_hour":          "R07",
    "tx_dayofweek":     "R07",
    "DeviceType":       "R05",
    "DeviceInfo":       "R05",
}

# Regex-based fallback for feature groups
_GROUP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^C\d+$"),    "R02"),   # C-features: card/account counts (velocity)
    (re.compile(r"^D\d+$"),    "R06"),   # D-features: time deltas (recency)
    (re.compile(r"^M\d+$"),    "R08"),   # M-features: match flags (identity)
    (re.compile(r"^id_0[1-9]$|^id_1[0-9]$|^id_2[0-4]$"), "R10"),  # id_ numeric (identity)
    (re.compile(r"^id_2[5-9]$|^id_3[0-8]$"), "R05"),               # id_ device fields
    (re.compile(r"^V\d+$"),    "R12"),   # V-features: Vesta proprietary signals
]


def feature_to_reason_code(feature_name: str) -> str:
    """Return the ECOA-style reason code for a given feature name."""
    if feature_name in _FEATURE_CODE_MAP:
        return _FEATURE_CODE_MAP[feature_name]
    for pattern, code in _GROUP_PATTERNS:
        if pattern.match(feature_name):
            return code
    return "R12"  # catch-all: aggregate risk signal


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def build_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Build a TreeExplainer for the champion model."""
    return shap.TreeExplainer(model)


def compute_shap(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    sample_n: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for X (or a random sample of size sample_n).

    Returns
    -------
    shap_values : ndarray of shape (n_rows, n_features)
    X_used      : the DataFrame rows that were explained (may be a sample)
    """
    if sample_n is not None and sample_n < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_n, replace=False)
        X_used = X.iloc[idx].reset_index(drop=True)
    else:
        X_used = X.reset_index(drop=True)

    explainer = build_explainer(model)
    shap_values = explainer.shap_values(X_used)
    logger.info("SHAP values computed — shape: %s", shap_values.shape)
    return shap_values, X_used


# ---------------------------------------------------------------------------
# Per-transaction reason codes
# ---------------------------------------------------------------------------

def get_top_reasons(
    transaction_id: int,
    explainer: shap.TreeExplainer,
    df: pd.DataFrame,
    feature_cols: list[str],
    n: int = 4,
) -> list[dict]:
    """Return top-n SHAP feature contributions for a single transaction.

    Parameters
    ----------
    transaction_id : value in df['TransactionID']
    explainer      : pre-built TreeExplainer (reuse across calls for speed)
    df             : DataFrame containing TransactionID and feature_cols
    feature_cols   : ordered list of model feature names
    n              : number of top reasons to return

    Returns
    -------
    List of dicts, each with keys:
        rank, feature, shap_value, direction, reason_code, reason_text
    """
    mask = df["TransactionID"] == transaction_id
    if not mask.any():
        raise ValueError(f"TransactionID {transaction_id!r} not found in provided DataFrame")

    X_row = df.loc[mask, feature_cols].head(1)
    shap_vals = explainer.shap_values(X_row)[0]  # shape (n_features,)

    # Rank by absolute contribution
    abs_order = np.argsort(np.abs(shap_vals))[::-1][:n]

    reasons = []
    for rank, idx in enumerate(abs_order, start=1):
        feat = feature_cols[idx]
        sv = float(shap_vals[idx])
        code = feature_to_reason_code(feat)
        reasons.append({
            "rank": rank,
            "feature": feat,
            "feature_value": float(X_row.iloc[0, idx]) if not pd.isna(X_row.iloc[0, idx]) else None,
            "shap_value": round(sv, 5),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
            "reason_code": code,
            "reason_text": REASON_CODES[code],
        })
    return reasons


def to_reason_codes(top_reasons: list[dict]) -> list[str]:
    """Extract deduplicated reason codes from get_top_reasons output, ordered by rank."""
    seen: set[str] = set()
    codes: list[str] = []
    for r in top_reasons:
        if r["reason_code"] not in seen:
            seen.add(r["reason_code"])
            codes.append(r["reason_code"])
    return codes


# ---------------------------------------------------------------------------
# Global importance summary (for notebooks / model docs)
# ---------------------------------------------------------------------------

def mean_abs_shap(shap_values: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame of features ranked by mean |SHAP| value."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_shap_arrays(
    shap_values: np.ndarray,
    X_used: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Persist SHAP values and corresponding X rows for notebook reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "shap_values.npy", shap_values)
    X_used.to_parquet(output_dir / "shap_X_sample.parquet", index=False)
    logger.info("SHAP arrays saved → %s", output_dir)
