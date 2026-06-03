"""Compute SHAP values and save summary plot + arrays. Run after train.py.

Usage:
    uv run python scripts/phase2_shap.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import shap

from src.models.champion import load_champion
from src.models.train import load_splits
from src.explain.shap_utils import (
    compute_shap, mean_abs_shap, save_shap_arrays,
    feature_to_reason_code, REASON_CODES,
)

RESULTS_DIR = Path("docs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model and data …")
model = load_champion()
_, _, X_val, y_val, feature_cols = load_splits(Path("data/processed"))

print("Computing SHAP values on 3,000-row sample …")
shap_values, X_sample = compute_shap(model, X_val, sample_n=3000, random_state=42)
save_shap_arrays(shap_values, X_sample)

# --- Summary plot ---
print("Generating SHAP summary plot …")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, max_display=20, show=False, plot_size=(10, 8))
plt.title("SHAP Summary — Top 20 Features (n=3,000 val sample)", pad=12)
plt.tight_layout()
out = RESULTS_DIR / "shap_summary.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved → {out}")

# --- Importance table ---
importance_df = mean_abs_shap(shap_values, feature_cols)
top20 = importance_df.head(20).copy()
top20["reason_code"] = top20["feature"].map(feature_to_reason_code)
top20["reason_text"] = top20["reason_code"].map(REASON_CODES)

print("\nTop 20 features by mean |SHAP|:")
print(f"  {'Feature':<22} {'Mean |SHAP|':>12}  Code  Description")
print("  " + "-" * 75)
for _, row in top20.iterrows():
    print(f"  {row['feature']:<22} {row['mean_abs_shap']:>12.5f}  {row['reason_code']}   {row['reason_text']}")
