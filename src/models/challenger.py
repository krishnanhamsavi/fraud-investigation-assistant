"""Challenger models for SR 11-7 benchmarking against the champion XGBoost.

SR 11-7 rationale
-----------------
The Federal Reserve's SR 11-7 guidance requires that model risk be managed through
independent validation, which includes benchmarking the proposed model against
conceptually simpler alternatives. Two challengers are defined:

  Challenger A — Logistic Regression
      The "simple, interpretable baseline." Uses a curated 35-feature set with
      target encoding and standard scaling. Intentionally excludes the anonymised
      V-features: feeding opaque inputs to a linear model produces neither
      interpretability nor predictive power, defeating the point of the exercise.

  Challenger B — LightGBM
      An alternative gradient boosting implementation with different structural
      hyperparameters (leaf-wise growth vs XGBoost's depth-wise) and a
      missingness-filtered feature set. Tests whether the champion's edge is
      robust across GBM implementations.
"""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Challenger A — Logistic Regression
# ---------------------------------------------------------------------------

# Interpretable feature set: named, domain-meaningful features only.
# The C-features (count) capture velocity; D-features capture recency;
# M-features capture identity match flags. No anonymised V-features.

LR_CAT_FEATURES: list[str] = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M4", "M5", "M6",
    "uid_card1_email",
]

LR_NUM_FEATURES: list[str] = [
    "TransactionAmt", "TransactionAmt_log",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1",
    "tx_hour", "tx_dayofweek",
    "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C13", "C14",
    "D1", "D2", "D3", "D4", "D10", "D15",
]

LR_ALL_FEATURES: list[str] = LR_NUM_FEATURES + LR_CAT_FEATURES


def _prep_lr(df: pd.DataFrame) -> pd.DataFrame:
    """Select LR features; coerce categorical columns to object for TargetEncoder."""
    available = [c for c in LR_ALL_FEATURES if c in df.columns]
    out = df[available].copy()
    for c in LR_CAT_FEATURES:
        if c in out.columns:
            out[c] = out[c].astype(object)
    return out


def build_lr_pipeline() -> Pipeline:
    """Logistic Regression pipeline: target-encode cats, median-impute + scale nums."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                LR_NUM_FEATURES,
            ),
            (
                "cat",
                # TargetEncoder (sklearn ≥1.3) uses k-fold cross-encoding on the
                # training set to prevent within-fold target leakage.
                TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42),
                LR_CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=0.1,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_challenger_a(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: Path = MODEL_DIR / "challenger_a_lr.joblib",
) -> tuple[Pipeline, list[str]]:
    """Fit Challenger A on the raw (pre-ordinal-encoded) DataFrame.

    X_train must contain the original string-valued categorical columns —
    pass data loaded from joined.parquet + build_features(), not train.parquet.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    X = _prep_lr(X_train)
    feature_cols = X.columns.tolist()

    missing = set(LR_ALL_FEATURES) - set(feature_cols)
    if missing:
        logger.warning("LR: %d requested features absent from data — skipped: %s", len(missing), missing)

    logger.info(
        "Training Challenger A (Logistic Regression) — %d features on %d rows",
        len(feature_cols), len(X),
    )
    pipe = build_lr_pipeline()
    pipe.fit(X, y_train)

    joblib.dump({"model": pipe, "feature_cols": feature_cols}, model_path)
    logger.info("Challenger A saved → %s", model_path)
    return pipe, feature_cols


def predict_lr(pipe: Pipeline, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return pipe.predict_proba(_prep_lr(X)[feature_cols])[:, 1]


def load_challenger_a(
    model_path: Path = MODEL_DIR / "challenger_a_lr.joblib",
) -> tuple[Pipeline, list[str]]:
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


# ---------------------------------------------------------------------------
# Challenger B — LightGBM
# ---------------------------------------------------------------------------

# Drop features that are >95% missing — they add noise to split-finding in
# leaf-wise trees without contributing signal. XGBoost is more robust to these
# via depth-wise growth; this is an intentional architectural difference.
LGBM_MISSING_THRESHOLD: float = 0.95

LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,          # leaf-wise (vs XGBoost depth-wise max_depth=6 → 63 leaves)
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "feature_fraction": 0.7,   # LightGBM's colsample equivalent
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def get_lgbm_features(X_train: pd.DataFrame) -> list[str]:
    """Return columns with ≤LGBM_MISSING_THRESHOLD missingness (computed on train only)."""
    miss = X_train.isnull().mean()
    kept = miss[miss <= LGBM_MISSING_THRESHOLD].index.tolist()
    logger.info(
        "LightGBM feature filter: %d kept, %d dropped (>%.0f%% missing)",
        len(kept), len(X_train.columns) - len(kept),
        LGBM_MISSING_THRESHOLD * 100,
    )
    return kept


def train_challenger_b(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_path: Path = MODEL_DIR / "challenger_b_lgbm.txt",
) -> tuple[lgb.LGBMClassifier, list[str]]:
    """Fit Challenger B on the processed (ordinal-encoded) DataFrame."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    features = get_lgbm_features(X_train)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)

    logger.info(
        "Training Challenger B (LightGBM) — %d features, neg/pos=%.1f",
        len(features), spw,
    )
    model.fit(
        X_train[features], y_train,
        eval_set=[(X_val[features], y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    model.booster_.save_model(str(model_path))
    # Persist feature list alongside model for cache-load reproducibility
    feat_path = model_path.with_suffix(".features.json")
    import json
    feat_path.write_text(json.dumps(features))
    logger.info("Challenger B saved → %s (%d features)", model_path, len(features))
    return model, features


def load_challenger_b(
    model_path: Path = MODEL_DIR / "challenger_b_lgbm.txt",
) -> tuple[lgb.Booster, list[str]]:
    import json
    feat_path = model_path.with_suffix(".features.json")
    features = json.loads(feat_path.read_text())
    booster = lgb.Booster(model_file=str(model_path))
    return booster, features
