# Model Monitoring Plan

**Document type:** Model Risk Management — Ongoing Monitoring Plan  
**Regulatory reference:** SR 11-7 Section V — Ongoing Monitoring  
**Review cycle:** Monthly (or triggered by alert conditions below)

---

## 1. Overview

SR 11-7 requires that models in use be subject to ongoing performance monitoring.
This plan defines what is monitored, at what frequency, and what thresholds trigger
escalation or retraining. All monitoring outputs are produced by
`scripts/phase4_monitoring.py` and stored in `docs/results/`.

---

## 2. Score Distribution Monitoring

The model's output score distribution is the primary monitoring signal. Drift in
the score distribution indicates that the input population has changed relative to
the training population — even before performance metrics degrade.

### 2.1 PSI (Population Stability Index) on Fraud Scores

| PSI Value | Classification | Action |
|-----------|----------------|--------|
| < 0.10 | Stable | No action |
| 0.10 – 0.20 | Moderate drift | Increase monitoring frequency; investigate top-drifting features |
| > 0.20 | Significant drift | Escalate to model risk team; trigger retraining review within 30 days |

**Reference period:** Training set score distribution (stored at model deployment).  
**Monitoring period:** Most recent 30-day window of production scores.

### 2.2 Kolmogorov-Smirnov Test on Fraud Scores

The KS test is run alongside PSI as a complementary check. PSI is sensitive to
distributional shape; KS is sensitive to any distributional difference.

| Condition | Action |
|-----------|--------|
| KS stat > 0.05 AND p < 0.05 | Flag for investigation |
| KS stat > 0.10 AND p < 0.01 | Escalate alongside PSI |

---

## 3. Feature-Level Drift Monitoring

Feature PSI is computed for the top 30 features by SHAP importance. This identifies
*which* input changes are driving score distribution shift — critical for root cause
analysis when the score PSI triggers an alert.

### 3.1 Feature PSI Thresholds

Same thresholds as score PSI (< 0.10 stable, 0.10–0.20 moderate, > 0.20 significant).

**Priority features** (top 5 by SHAP importance — C13, card6, TransactionAmt, C14, V70):
- Significant drift in ANY of these 5 triggers an immediate alert regardless of
  aggregate score PSI, because these features drive the majority of model output.

### 3.2 Missingness Monitoring

Separately from value distribution, the **missing rate** of each feature is tracked.
An increase in missingness on a high-importance feature may indicate an upstream
data pipeline issue, not genuine population drift.

| Change in missingness | Action |
|----------------------|--------|
| ±2 percentage points on top-5 feature | Investigate data pipeline |
| ±5 percentage points on any top-20 feature | Alert and investigate |

---

## 4. Performance Metric Monitoring

When labelled data becomes available (fraud confirmed / cleared through operations),
model performance is recomputed on a rolling 30-day window.

### 4.1 Primary Metrics and Thresholds

| Metric | Deployment Value | Alert Threshold | Retraining Threshold |
|--------|-----------------|-----------------|---------------------|
| ROC-AUC | 0.9126 | < 0.900 (−0.013) | < 0.880 (−0.033) |
| PR-AUC | 0.5291 | < 0.480 (−0.049) | < 0.450 (−0.079) |
| FDR @ 3% review | 47.83% | < 42% (−5.8 ppt) | < 38% (−9.8 ppt) |
| FDR @ 5% review | 58.10% | < 53% (−5.1 ppt) | < 48% (−10.1 ppt) |

Thresholds are set at approximately 1.5× and 3× the estimated sampling noise for
a 30-day window. Tighter thresholds would produce too many false alarms;
looser thresholds would delay necessary action.

### 4.2 Label Lag Consideration

Fraud labels typically arrive with a 30–90 day lag (chargeback cycle). Performance
monitoring therefore operates on a delayed basis. Score distribution monitoring
(PSI/KS) serves as the early-warning system while labels are pending.

---

## 5. Fairness Metric Monitoring

Fairness metrics (disparate impact ratio, FPR gap) are recomputed in each monthly
monitoring cycle alongside performance metrics.

| Condition | Action |
|-----------|--------|
| DI ratio drops below 0.80 (4/5ths rule) | Immediate escalation to Fair Lending Officer |
| FPR gap increases by > 5 ppt vs deployment | Escalate to model risk team |
| DI ratio 0.80–0.85 (approaching threshold) | Increase to bi-weekly monitoring |

---

## 6. Retraining Decision Framework

Retraining is not automatic. All retraining triggers initiate a **review** process.

```
Alert triggered
     │
     ▼
Root cause analysis (≤ 5 business days)
     │
     ├── Data pipeline issue → fix pipeline, no retrain needed
     │
     ├── Temporary anomaly → watchlist, monitor next cycle
     │
     └── Genuine drift → Retraining justified
              │
              ▼
         Retrain champion on updated data
              │
              ▼
         Full validation against challengers (repeat Phase 3)
              │
              ▼
         SR 11-7 model change review (material vs non-material)
              │
              ▼
         Deploy updated model
```

### 6.1 Material vs Non-Material Change Classification

Per SR 11-7, model changes are classified as material (requiring full re-validation)
or non-material (requiring expedited review):

| Change | Classification |
|--------|----------------|
| Retraining on expanded dataset, same architecture | Non-material |
| Hyperparameter change with < 2% AUC impact | Non-material |
| Addition of new feature group | Material |
| Architecture change (e.g. switch to neural network) | Material |
| Change to scoring output format or threshold | Material |

---

## 7. Monitoring Schedule

| Activity | Frequency | Owner | Output |
|----------|-----------|-------|--------|
| Score PSI + KS test | Monthly | Model Developer | `score_drift.json` |
| Feature PSI (top 30) | Monthly | Model Developer | `feature_psi.csv` |
| Performance metrics | Monthly (30-day lag) | Model Validator | Updated metrics JSON |
| Fairness metrics | Monthly | Fair Lending / MRM | `fairness_results.json` |
| Full validation review | Annually or on trigger | Independent Validator | Validation report |
| Challenger benchmarking | Annually | Model Developer | Updated `02_challenger_analysis.md` |

---

## 8. Escalation Matrix

| Severity | Condition | Escalation Path | SLA |
|----------|-----------|-----------------|-----|
| P1 — Critical | Score PSI > 0.30 OR AUC < 0.860 | Model Risk Officer → CRO | 24 hours |
| P2 — High | Score PSI > 0.20 OR AUC < 0.880 | Model Risk Team | 5 business days |
| P3 — Medium | Score PSI > 0.10 OR feature drift in top-5 | Model Developer | 15 business days |
| P4 — Low | Moderate drift in non-priority features | Monitor next cycle | Next cycle |

---

## 9. Tools and Reproducibility

All monitoring computations are implemented in:
- `src/monitoring/drift.py` — PSI, KS test, feature drift report
- `src/monitoring/fairness.py` — disparate impact analysis
- `scripts/phase4_monitoring.py` — end-to-end runner

Results are deterministic given the same input data. The monitoring script can be
run by any analyst with access to the processed data and model artifacts, without
requiring the original model developer.
