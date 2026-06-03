"""Feature engineering and time-based train/validation split for IEEE-CIS data."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.data.load import load_raw

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# Columns that are identifiers or targets — never used as model features
_DROP_FROM_FEATURES = {"TransactionID", "isFraud"}

# Object-dtype columns whose label-encoding should be fit on train only
# (detected dynamically in encode_categoricals, but listed here for documentation)
_KNOWN_CAT_COLS = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features in-place (works on any split or the full frame).

    New columns
    -----------
    TransactionAmt_log  : log1p of TransactionAmt — compresses the right tail.
    uid_card1_email     : composite key (card1 + P_emaildomain); captures
                          per-card / per-domain behaviour without aggregation.
    tx_hour             : hour of day derived from TransactionDT (0–24).
    tx_dayofweek        : day-of-week proxy (0–6) from TransactionDT.
    """
    df = df.copy()

    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # UID feature: card identity × email domain
    df["uid_card1_email"] = (
        df["card1"].astype(str) + "_" + df["P_emaildomain"].fillna("nan")
    )

    # Time features — TransactionDT is seconds elapsed from an undisclosed epoch
    df["tx_hour"] = (df["TransactionDT"] % 86400) / 3600          # 0–24
    df["tx_dayofweek"] = (df["TransactionDT"] // 86400) % 7       # 0–6

    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by TransactionDT; first (1-val_frac) → train, last val_frac → val.

    No row is shared between splits, and max(train.DT) <= min(val.DT) is
    guaranteed — there is no temporal leakage.
    """
    df_sorted = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_frac))
    train = df_sorted.iloc[:split_idx].copy()
    val = df_sorted.iloc[split_idx:].copy()
    logger.info(
        "Split → train: %d rows, val: %d rows  (boundary DT %d / %d)",
        len(train), len(val),
        train["TransactionDT"].max(), val["TransactionDT"].min(),
    )
    return train, val


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder, list[str]]:
    """Label-encode all object-dtype columns using OrdinalEncoder fit on train.

    Categories unseen in training (including the engineered uid feature) are
    encoded as NaN so XGBoost treats them as missing — no data leakage.

    Returns
    -------
    train_enc, val_enc : encoded copies of the input frames
    encoder            : fitted OrdinalEncoder (for inference-time use)
    cat_cols           : list of columns that were encoded
    """
    # pandas 3.0 infers string columns as StringDtype (not object), so we
    # cannot rely on dtype == object. Exclude numeric/bool; keep everything else.
    cat_cols = [
        c for c in train.columns
        if not pd.api.types.is_numeric_dtype(train[c])
        and not pd.api.types.is_bool_dtype(train[c])
        and c not in _DROP_FROM_FEATURES
    ]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
        dtype=np.float64,
    )

    train = train.copy()
    val = val.copy()

    # Coerce to object so sklearn handles StringDtype consistently
    for c in cat_cols:
        train[c] = train[c].astype(object)
        val[c] = val[c].astype(object)

    train[cat_cols] = encoder.fit_transform(train[cat_cols])
    val[cat_cols] = encoder.transform(val[cat_cols])

    logger.info("Encoded %d categorical columns.", len(cat_cols))
    return train, val, encoder, cat_cols


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = PROCESSED_DIR,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end preprocessing: load → feature engineering → split → encode → save.

    Outputs written to output_dir
    ------------------------------
    train.parquet       : encoded training features + isFraud target
    val.parquet         : encoded validation features + isFraud target
    preprocessor.joblib : dict with keys 'encoder' and 'cat_cols'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(data_dir)
    df = build_features(df)
    train, val = time_split(df, val_frac=val_frac)
    train, val, encoder, cat_cols = encode_categoricals(train, val)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    joblib.dump(
        {"encoder": encoder, "cat_cols": cat_cols},
        output_dir / "preprocessor.joblib",
    )

    fraud_train = train["isFraud"].mean()
    fraud_val = val["isFraud"].mean()
    logger.info(
        "Fraud rate — train: %.4f (%.1f%%), val: %.4f (%.1f%%)",
        fraud_train, fraud_train * 100,
        fraud_val, fraud_val * 100,
    )
    return train, val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_preprocessing()
