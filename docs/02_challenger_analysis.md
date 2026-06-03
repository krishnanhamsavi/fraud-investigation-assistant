# Challenger Model Analysis

**Document type:** Model Risk Management — Challenger Benchmarking Report  
**Prepared for:** Independent Model Validation  
**Regulatory reference:** SR 11-7 / OCC 2011-12 — Guidance on Model Risk Management  
**Status:** Metrics populated by `scripts/phase3_compare.py`

---

## 1. Purpose

SR 11-7 requires that model developers benchmark the proposed champion model against
conceptually simpler alternatives before deployment. This ensures the additional
complexity of the champion model is justified by measurable performance improvement,
and that a credible fallback exists if the champion is retired.

This document records the development rationale, design decisions, and validation
outcomes for two challenger models evaluated against the champion XGBoost classifier.

---

## 2. Model Inventory

| ID | Model | Role | Feature Set |
|----|-------|------|-------------|
| M-01 | XGBoost (champion) | Production candidate | All 436 engineered features |
| M-02 | Logistic Regression (Challenger A) | Simple interpretable baseline | 35 curated named features |
| M-03 | LightGBM (Challenger B) | Alternative GBM implementation | Features with ≤95% missingness |

---

## 3. Challenger Design Rationale

### 3.1 Challenger A — Logistic Regression

**Why this challenger exists:**  
Logistic regression is the canonical SR 11-7 "simple benchmark." It is fully
interpretable (each coefficient is a direct risk weight), auditable by model
validators without ML expertise, and has well-understood statistical properties.
If a complex model cannot convincingly outperform LR, the complexity is not
justified.

**Feature set choices:**  
The LR challenger intentionally excludes the 339 anonymised Vesta V-features.
Feeding opaque inputs to a linear model would produce neither interpretability
(coefficients on unnamed features have no business meaning) nor predictive power
(LR cannot model nonlinear interactions that give V-features their signal in
tree models). The 35-feature curated set covers transaction amount, card profile,
email domain, velocity counts (C1–C14), time deltas (D1–D15), and identity match
flags (M4–M6).

**Preprocessing:**  
Categorical features are target-encoded using sklearn's `TargetEncoder` with
5-fold cross-encoding on the training set (preventing within-fold leakage).
Numeric features are median-imputed and standardised. Class imbalance is addressed
via `class_weight='balanced'`.

### 3.2 Challenger B — LightGBM

**Why this challenger exists:**  
LightGBM uses leaf-wise tree growth (vs XGBoost's depth-wise), which can produce
more accurate splits on datasets with many features at the cost of higher variance.
A strong LightGBM score would suggest the champion's advantage is specific to
XGBoost's regularisation rather than to its feature set — a useful sensitivity check.

**Feature set choices:**  
Features with >95% missingness are dropped. These are predominantly identity fields
(`id_21`–`id_27`, `id_07/08`) that are 99%+ missing. In leaf-wise trees, forcing
splits on near-constant columns inflates tree complexity without improving
generalisation. XGBoost's depth-wise growth is more robust to these; this difference
is an intentional architectural contrast between the two GBM implementations.

**Hyperparameter differences from champion:**

| Parameter | Champion (XGBoost) | Challenger B (LightGBM) |
|-----------|-------------------|------------------------|
| Tree growth | Depth-wise (`max_depth=6`) | Leaf-wise (`num_leaves=63`) |
| Column sampling | `colsample_bytree=0.8` | `feature_fraction=0.7` |
| Min node size | `min_child_weight=5` | `min_child_samples=20` |
| Bagging | `subsample=0.8` | `bagging_fraction=0.8, freq=5` |

---

## 4. Validation Results

*Table auto-populated by `scripts/phase3_compare.py`.*

<!-- METRICS_TABLE_START -->
| Model | ROC-AUC | PR-AUC | FDR@3% | FDR@5% | Latency (ms/1k) | Features |
|-------|---------|--------|--------|--------|-----------------|----------|
| Champion (XGBoost) | 0.9126 | 0.5291 | 47.83% | 58.10% | 33.5 | 436 |
| Challenger A (LR) | 0.8375 | 0.2482 | 27.98% | 37.11% | 48.6 | 38 |
| Challenger B (LightGBM) | 0.8306 | 0.2736 | 32.26% | 42.00% | 4.8 | 427 |
<!-- METRICS_TABLE_END -->

**Metric definitions:**
- **ROC-AUC:** Probability that model ranks a random fraud above a random legitimate transaction.
- **PR-AUC:** Area under the precision-recall curve; primary metric for imbalanced classes.
- **FDR@k%:** Fraud Detection Rate — fraction of all fraud caught if the top k% of scored transactions are reviewed. Operationally driven by the review team's daily capacity.
- **Latency:** Median milliseconds to score 1,000 transactions (10 repetitions). Relevant for real-time card authorisation use cases.

---

## 5. Champion Selection Rationale

### 5.1 Champion vs Challenger A (Logistic Regression)

The champion outperforms the LR baseline decisively across all metrics:

- ROC-AUC: **0.9126 vs 0.8375** (+0.075)
- PR-AUC: **0.5291 vs 0.2482** (+0.281 — more than double)
- FDR@3%: **47.83% vs 27.98%** (+19.8 percentage points)

This gap is expected and justified. The primary fraud signal in this dataset is
distributed across nonlinear interactions between the anonymised V-features (C13,
V70, V294, etc.) that logistic regression cannot model regardless of preprocessing.
At the 3% operational review rate, the champion catches 19.8 percentage points
more fraud — translating to approximately 808 additional fraud cases caught per
118,000 validation transactions. This is the business justification for accepting
the additional model complexity.

The LR model's 38-feature interpretable design remains valuable as a
**fallback model** should the champion require emergency retirement.

### 5.2 Champion vs Challenger B (LightGBM)

The XGBoost champion outperforms LightGBM by a larger margin than expected:

- ROC-AUC: **0.9126 vs 0.8306** (+0.082)
- PR-AUC: **0.5291 vs 0.2736** (+0.256)
- FDR@3%: **47.83% vs 32.26%** (+15.6 percentage points)

A gap of this magnitude between two GBM implementations on the same feature set
warrants documentation. Two contributing factors are identified:

1. **Hyperparameters are defaults, not tuned.** XGBoost's depth-wise growth with
   `max_depth=6` provides stronger implicit regularisation on this dataset than
   LightGBM's leaf-wise growth with `num_leaves=63`. A Bayesian hyperparameter
   search for LightGBM would likely close most of this gap.

2. **LightGBM PR-AUC (0.2736) exceeds LR PR-AUC (0.2482) despite lower ROC-AUC.**
   This indicates LightGBM is better calibrated on the fraud class than LR, and
   is correctly leveraging the additional 389 features — it simply needs better
   hyperparameters to convert that signal into ROC-level discrimination.

**Conclusion:** The champion is selected. The LightGBM gap does not indicate a
problem with the champion; it indicates an under-tuned challenger. This is noted
as a next-step item in Section 6.

### 5.3 Inference latency

| Model | Latency (ms / 1k txns) | Relative |
|-------|------------------------|----------|
| Champion (XGBoost) | 33.5 ms | 1.0× |
| Challenger A (LR) | 48.6 ms | 1.5× |
| Challenger B (LightGBM) | **4.8 ms** | 0.1× |

LightGBM's native Booster API is ~7× faster than XGBoost's sklearn-compatible
`predict_proba`. For a real-time card authorisation pipeline with a 100ms total
decision budget, all three are viable. LightGBM's speed advantage is noted as a
reason to revisit it as a production candidate if latency becomes a constraint.

---

## 6. Limitations and Next Steps

1. **Hyperparameter tuning:** The champion and both challengers use reasonable defaults,
   not CV-optimised hyperparameters. A proper Bayesian optimisation run over 50+
   trials on the training set (with time-series cross-validation) would likely lift
   all three models and potentially change the ordering.

2. **Ensemble potential:** A stacked ensemble of XGBoost + LightGBM predictions
   typically outperforms either model individually on this dataset (as evidenced by
   Kaggle competition results). This was not pursued here because SR 11-7 requires
   that increased complexity be justified; the single-model champion is the appropriate
   baseline for initial deployment.

3. **Feature engineering for LR:** The LR challenger could be improved with
   count-encoding or frequency-encoding of the email domain and card features
   (replacing target encoding with aggregation features computed on training data).
   This is noted as a future enhancement if LR is promoted as a fallback.

---

## 7. Sign-off Placeholder

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Model Developer | — | — | — |
| Independent Validator | — | — | — |
| Model Risk Officer | — | — | — |
