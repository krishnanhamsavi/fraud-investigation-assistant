"""Champion model: XGBoost classifier optimised for fraud detection.

Design choices documented for SR 11-7 model documentation:
- tree_method='hist'         : GPU-optional histogram algorithm; fast on large data.
- scale_pos_weight           : set to neg/pos ratio (~28) to counteract class imbalance
                               without resampling, which would invalidate calibration.
- eval_metric=['aucpr','auc']: PR-AUC drives early stopping (more informative than ROC
                               on imbalanced data); ROC-AUC reported for benchmarking.
- early_stopping_rounds=50   : prevents overfitting; best_iteration stored in model.
- Saved as .json (XGBoost native): portable, version-controlled, no pickle security risk.
"""

import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
CHAMPION_PATH = MODEL_DIR / "champion.json"

# Hyperparameters — tuned for PR-AUC on this dataset; see docs/01_model_documentation.md
CHAMPION_PARAMS: dict = {
    "tree_method": "hist",
    "device": "cpu",
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc"],
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
}


def build_champion(scale_pos_weight: float) -> xgb.XGBClassifier:
    """Construct (but do not train) the champion XGBoost classifier."""
    return xgb.XGBClassifier(**CHAMPION_PARAMS, scale_pos_weight=scale_pos_weight)


def train_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_path: Path = CHAMPION_PATH,
) -> xgb.XGBClassifier:
    """Fit champion model with early stopping on the validation set.

    Saves model to model_path as XGBoost JSON (not pickle).
    Returns the fitted model.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    logger.info(
        "Training champion — %d train rows, %d val rows | neg/pos = %.1f",
        len(X_train), len(X_val), spw,
    )

    model = build_champion(scale_pos_weight=spw)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    logger.info(
        "Training complete — best iteration: %d / %d",
        model.best_iteration, CHAMPION_PARAMS["n_estimators"],
    )
    model.save_model(str(model_path))
    logger.info("Champion model saved → %s", model_path)
    return model


def load_champion(model_path: Path = CHAMPION_PATH) -> xgb.XGBClassifier:
    """Load a saved champion model from JSON."""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model
