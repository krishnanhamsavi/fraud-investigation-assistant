"""Run after Phase 1 preprocessing — prints the stats needed for the phase report.

Usage (from project root):
    python scripts/phase1_stats.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import pandas as pd
from src.data.preprocess import run_preprocessing

print("=" * 60)
print("Running preprocessing pipeline …")
print("=" * 60)

train, val = run_preprocessing()

print("\n" + "=" * 60)
print("PHASE 1 REPORT STATS")
print("=" * 60)

print(f"\nRow counts")
print(f"  Train : {len(train):>9,}")
print(f"  Val   : {len(val):>9,}")
print(f"  Total : {len(train)+len(val):>9,}")

print(f"\nClass balance")
print(f"  Train fraud rate : {train['isFraud'].mean():.4%}  ({train['isFraud'].sum():,} fraud rows)")
print(f"  Val   fraud rate : {val['isFraud'].mean():.4%}  ({val['isFraud'].sum():,} fraud rows)")

print(f"\nTemporal boundary check (no leakage)")
print(f"  Max train TransactionDT : {train['TransactionDT'].max():,}")
print(f"  Min val   TransactionDT : {val['TransactionDT'].min():,}")
print(f"  Leakage-free            : {train['TransactionDT'].max() <= val['TransactionDT'].min()}")

print(f"\nTop 10 columns by missingness (val split)")
null_pct = val.isnull().mean().sort_values(ascending=False).head(10)
for col, pct in null_pct.items():
    print(f"  {col:<20s}  {pct:.1%}")

print(f"\nFeature count : {train.shape[1] - 2} model features + isFraud + TransactionID")
print("=" * 60)
