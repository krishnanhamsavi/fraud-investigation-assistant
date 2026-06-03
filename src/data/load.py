"""Load and join IEEE-CIS transaction and identity tables."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def load_raw(data_dir: str | Path = RAW_DIR, *, force: bool = False) -> pd.DataFrame:
    """Load train_transaction.csv + train_identity.csv, left-join on TransactionID.

    Caches the result as data/processed/joined.parquet on first run.
    Subsequent calls return the cached file unless force=True.

    Parameters
    ----------
    data_dir : path to the directory containing the raw Kaggle CSVs.
    force    : if True, re-read from CSV and overwrite the cache.
    """
    data_dir = Path(data_dir)
    cache_path = PROCESSED_DIR / "joined.parquet"

    if cache_path.exists() and not force:
        logger.info("Loading cached joined data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Reading train_transaction.csv (~590 k rows) …")
    transactions = pd.read_csv(data_dir / "train_transaction.csv")

    logger.info("Reading train_identity.csv …")
    identity = pd.read_csv(data_dir / "train_identity.csv")

    logger.info("Left-joining on TransactionID …")
    df = transactions.merge(identity, on="TransactionID", how="left")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(
        "Saved joined data → %s  (%d rows × %d cols)",
        cache_path, len(df), df.shape[1],
    )
    return df
