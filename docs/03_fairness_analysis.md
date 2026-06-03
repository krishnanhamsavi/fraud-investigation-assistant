# Fairness Analysis

**Document type:** Model Risk Management — Fairness & Disparate Impact Assessment  
**Regulatory reference:** ECOA (Reg B), Fair Housing Act, CFPB Supervisory Guidance  
**Status:** Populated by `scripts/phase4_monitoring.py`

---

## 1. Important Limitation — Absence of Demographic Data

**The IEEE-CIS dataset contains no demographic information.** Transaction records
do not include customer race, color, sex, national origin, religion, marital status,
or age — the protected classes enumerated under ECOA and the Fair Housing Act.

This document therefore presents a **methodology demonstration** using proxy
variables available in the dataset. The analysis shows *how* a bank would conduct
fairness testing; it does not constitute a ECOA compliance assessment.

In a production deployment, this analysis would be repeated with:
- Demographic data from the core banking system (where legally permissible)
- BISG (Bayesian Improved Surname Geocoding) proxy if direct demographics unavailable
- Lending club or bureau-level data where applicable

---

## 2. Proxy Variables Selected

Three proxy variables are analysed. None are protected classes; they serve as
stand-ins to demonstrate the measurement methodology:

| Proxy | Description | Groups | Coverage |
|-------|-------------|--------|----------|
| `ProductCD` | Transaction product type (W/H/C/S/R) | 5 groups | ~100% |
| `card6` | Credit vs debit card type | 2 groups | ~95% |
| `DeviceType` | Desktop vs mobile device | 2 groups | ~20% |

`DeviceType` has high missingness (~80%) — results for this proxy should be
interpreted cautiously as the observed population may not be representative.

---

## 3. Analytical Framework

### 3.1 Threshold

Analysis is conducted at the **3% review rate threshold** — the operational
review capacity assumed in the champion model evaluation. This corresponds to
the score cutoff above which transactions are flagged for investigation.

### 3.2 Metrics

**Flag rate (adverse action rate):**  
The fraction of transactions from each group that are flagged.  
`P(flagged | group)`

**False positive rate (FPR):**  
The fraction of *legitimate* transactions that are incorrectly flagged.  
`P(flagged | legitimate, group)` — a measure of over-surveillance by group.

**True positive rate (TPR / recall):**  
The fraction of actual fraud caught within each group.  
`P(flagged | fraud, group)` — a measure of equal protection by group.

### 3.3 The 4/5ths Rule (80% Rule)

The EEOC's 4/5ths rule states: if the selection rate (here: flag rate) for any
group is less than 80% of the rate for the group with the **lowest** flag rate,
adverse impact is indicated. For fraud detection, lower flag rate = *less*
adverse treatment, so the rule asks: is the most-flagged group flagged more than
25% above the least-flagged group?

`DI ratio = min_group_flag_rate / max_group_flag_rate ≥ 0.80`

### 3.4 FPR Equality (Equalised Odds)

The 4/5ths rule only looks at aggregate flag rates. The **equalised odds**
criterion additionally requires that FPR (false alarm rate) be roughly equal
across groups — even if overall flag rates are similar, one group should not
bear a disproportionate share of false alarms.

---

## 4. Results

### 4.1 ProductCD

| Group | N | Actual Fraud Rate | Flag Rate | FPR | TPR |
|-------|---|------------------|-----------|-----|-----|
| C | 12,109 | 13.35% | 15.26% | 5.88% | 76.13% |
| H | 3,333 | 5.91% | 6.96% | 3.86% | 56.35% |
| R | 5,496 | 4.68% | 4.73% | 1.26% | 75.49% |
| S | 3,501 | 5.23% | 2.43% | 0.60% | 35.52% |
| W | 93,669 | 1.93% | 1.19% | 0.84% | 18.95% |

**DI ratio = 0.078 — adverse impact indicated by 4/5ths rule.**

The 13× flag rate difference (C vs W) is largely explained by a 7× actual fraud rate
difference. The model is responding proportionally to real risk. However, the FPR
gap of 5.28 ppt (product C at 5.88% vs S at 0.60%) means legitimate C-type customers
face nearly 10× the false alarm rate — an actionable equity finding worth monitoring.

### 4.2 card6 (credit vs debit)

| Group | N | Actual Fraud Rate | Flag Rate | FPR | TPR |
|-------|---|------------------|-----------|-----|-----|
| credit | 27,270 | 6.46% | 7.54% | 3.90% | 60.25% |
| debit | 90,095 | 2.54% | 1.62% | 0.67% | 38.16% |

**DI ratio = 0.215 — adverse impact indicated.**

Credit cards face 4.7× higher flag rate vs 2.5× higher actual fraud rate. Secondary
finding: debit card fraud recall is only 38.16% vs 60.25% for credit — the model
*under-protects* debit cardholders. Direct account-draining fraud receives less
model attention than credit fraud, worth noting to operations.

### 4.3 DeviceType

| Group | N | Actual Fraud Rate | Flag Rate | FPR | TPR |
|-------|---|------------------|-----------|-----|-----|
| mobile | 9,552 | 11.43% | 12.20% | 4.95% | 68.32% |
| desktop | 13,620 | 8.00% | 8.74% | 2.95% | 75.30% |
| missing | 94,936 | 1.98% | 1.25% | 0.87% | 20.07% |

**DI ratio = 0.103 — adverse impact indicated, but driven by base rate.**

The "missing" group (80% of transactions) has a 1.98% fraud rate and 20% TPR —
the model struggles to catch fraud without device signals. This is a model capability
gap, not a fairness issue. Increasing identity data capture would improve model
protection for the majority of transactions.

### 4.4 Key Drift Finding — TransactionDT

Feature PSI monitoring flagged `TransactionDT` at PSI = 18.8 ("significant drift").
This is **expected and non-actionable**: it is a monotonically increasing counter
across any time-based split. Its presence as rank-17 SHAP feature indicates the
model absorbed period-specific temporal patterns — a known limitation documented
in `docs/01_model_documentation.md` for annual validation review.

---

## 5. Mitigation Strategies Considered

### 5.1 If DI ratio < 0.80 is found

1. **Root cause analysis:** Determine whether the disparity is explained by
   legitimate risk differences (different fraud rates per group) or by feature
   correlation with a protected characteristic.

2. **Regularisation adjustment:** Increase model regularisation to reduce
   feature weights that correlate with proxy characteristics.

3. **Post-processing threshold adjustment:** Apply group-specific score thresholds
   to equalise FPR. This is an operational (not model) intervention and requires
   business and legal sign-off.

4. **Feature removal:** If a specific feature is driving the disparity and is a
   proxy for a protected class (e.g. billing zip code correlating with race),
   consider its removal or transformation.

### 5.2 Ongoing monitoring

Fairness metrics should be recomputed every monitoring cycle alongside PSI.
Flag rate disparities can increase over time as fraud patterns evolve and
model scores shift. A DI ratio that is 0.85 at deployment may drift to 0.78
within 6 months without performance degradation on aggregate metrics.

---

## 6. Limitations Summary

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No demographic data | Cannot assess ECOA compliance directly | BISG proxy if needed |
| Proxy variables not protected classes | Results are illustrative only | Link to core banking demographics |
| DeviceType 80% missing | Device fairness results are unrepresentative | Do not act on alone |
| Static threshold analysis | Fairness may differ at other operating points | Rerun at multiple thresholds |
| Single point in time | Fairness drift not measured | Include in monthly monitoring cycle |

---

## 7. Sign-off Placeholder

| Role | Name | Date |
|------|------|------|
| Model Developer | — | — |
| Fair Lending Officer | — | — |
| Independent Validator | — | — |
