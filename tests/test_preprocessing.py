"""Preprocessing pipeline tests — verifies correctness without touching raw data."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import build_features, encode_categoricals, time_split


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic transaction DataFrame that mimics the IEEE-CIS schema."""
    rng = np.random.default_rng(42)
    n = 2000

    # Monotonically increasing DT so we can verify temporal properties
    dt_base = rng.integers(86400, 86400 * 180, size=n)
    dt = np.sort(dt_base)

    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": dt,
            "TransactionAmt": rng.lognormal(4.0, 1.5, n),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
            "card1": rng.integers(1000, 9999, n),
            "card4": rng.choice(["visa", "mastercard", None], n),
            "card6": rng.choice(["credit", "debit", None], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "R_emaildomain": rng.choice(["hotmail.com", "outlook.com", None], n),
            "M1": rng.choice(["T", "F", None], n),
            "isFraud": rng.choice([0, 1], n, p=[0.965, 0.035]),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — output shapes and engineered feature presence
# ---------------------------------------------------------------------------

def test_output_shapes(sample_df):
    """train + val row counts sum to original; engineered columns are present."""
    df = build_features(sample_df)
    train, val = time_split(df, val_frac=0.2)

    assert len(train) + len(val) == len(df), "rows lost during split"
    assert set(train.columns) == set(val.columns), "column mismatch between splits"

    for col in ("TransactionAmt_log", "uid_card1_email", "tx_hour", "tx_dayofweek"):
        assert col in train.columns, f"engineered column '{col}' missing from train"


# ---------------------------------------------------------------------------
# Test 2 — no temporal leakage between train and val
# ---------------------------------------------------------------------------

def test_no_temporal_leakage(sample_df):
    """Every transaction in val must be at or after every transaction in train."""
    df = build_features(sample_df)
    train, val = time_split(df, val_frac=0.2)

    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), (
        "Temporal leakage: train contains transactions later than some val transactions"
    )


# ---------------------------------------------------------------------------
# Test 3 — temporal ordering preserved within each split
# ---------------------------------------------------------------------------

def test_time_ordering_preserved(sample_df):
    """Both splits must be sorted ascending by TransactionDT."""
    df = build_features(sample_df)
    train, val = time_split(df, val_frac=0.2)

    assert (train["TransactionDT"].diff().dropna() >= 0).all(), (
        "train split is not sorted by TransactionDT"
    )
    assert (val["TransactionDT"].diff().dropna() >= 0).all(), (
        "val split is not sorted by TransactionDT"
    )


# ---------------------------------------------------------------------------
# Test 4 — categorical encoding: unknown values become NaN, not errors
# ---------------------------------------------------------------------------

def test_unseen_categories_become_nan(sample_df):
    """Categories in val not seen during training must encode to NaN."""
    df = build_features(sample_df)
    train, val = time_split(df, val_frac=0.2)

    # Inject a novel category into the val split
    val = val.copy()
    val.iloc[0, val.columns.get_loc("ProductCD")] = "UNSEEN_XYZ"

    _, val_enc, _, cat_cols = encode_categoricals(train.copy(), val)

    assert pd.isna(val_enc.iloc[0]["ProductCD"]), (
        "Unseen category should encode to NaN so XGBoost treats it as missing"
    )


# ---------------------------------------------------------------------------
# Test 5 — split fraction is respected within ±1 row
# ---------------------------------------------------------------------------

def test_val_fraction(sample_df):
    """Val split should be approximately val_frac of the total rows."""
    df = build_features(sample_df)
    train, val = time_split(df, val_frac=0.2)

    actual_frac = len(val) / len(df)
    assert abs(actual_frac - 0.2) <= 1 / len(df) + 1e-9, (
        f"Val fraction {actual_frac:.4f} deviates from 0.2 by more than 1 row"
    )
