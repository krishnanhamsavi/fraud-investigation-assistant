# LLM Agent Governance Framework

**Document type:** Model Risk Management — GenAI System Governance  
**Regulatory reference:** SR 11-7 (extended to GenAI); OCC 2011-12; CFPB AI/ML Guidance  
**System:** Fraud Investigation Agent (Anthropic Claude Sonnet, tool-use architecture)  
**Status:** Complete

---

## 1. Overview and Regulatory Framing

SR 11-7 defines a model as "a quantitative method, system, or approach that applies
statistical, economic, financial, or mathematical theories, techniques, and assumptions
to process input data into quantitative estimates." Large language models used in
consequential decisions fit this definition: they process inputs (transaction data,
SHAP explanations) through learned statistical transformations to produce outputs
(risk assessments, recommended actions) that influence credit and fraud decisions.

This document applies SR 11-7's three-pillar framework — **development, validation,
and governance** — to the fraud investigation agent. Where SR 11-7 does not directly
address LLM-specific risks (hallucination, prompt injection, non-determinism), this
document extends the framework using NIST AI RMF and emerging industry practice.

**Critical principle:** The agent is a decision-support tool, not a decision-maker.
Every recommended action is presented to a human analyst for final determination.
This architectural constraint limits the agent's model risk classification to
"moderate" under SR 11-7 (consequential but not autonomous).

---

## 2. Agent Architecture

### 2.1 Components

| Component | Description | Governance implication |
|-----------|-------------|----------------------|
| Base LLM | Claude Sonnet (Anthropic) | Third-party model — vendor risk applies |
| System prompt | `src/agent/prompts/v1.txt` | Version-controlled; change requires review |
| Tool layer | 4 deterministic Python functions | Separate from LLM; independently testable |
| Guardrail layer | 4 automated checks post-generation | Acts as a safety net for LLM outputs |
| Fraud ML model | XGBoost champion (M-01) | Governed separately; feeds agent via tools |

### 2.2 Data flow

```
Flagged transaction (TransactionID)
         │
         ▼
    Agent receives task
         │
    ┌────▼────┐
    │  Tool   │  ← get_transaction_details  (deterministic lookup)
    │  calls  │  ← get_shap_explanation     (deterministic computation)
    │  (4×)   │  ← get_customer_history     (deterministic aggregation)
    │         │  ← get_reason_codes         (deterministic mapping)
    └────┬────┘
         │  Tool outputs injected into context
         ▼
    Claude generates structured JSON report
         │
         ▼
    Guardrails validate output
         │
         ▼
    Human analyst receives report + guardrail status
```

The tool layer is fully deterministic — given the same TransactionID and model state,
tools always return the same data. Non-determinism is **isolated to the LLM synthesis
step**, which is the component where hallucination, reasoning errors, and prompt
sensitivity risks reside.

---

## 3. Prompt Governance

### 3.1 Version control

System prompts are stored in `src/agent/prompts/` with semantic versioning:
- `v1.txt` — baseline prompt (this release)
- Future versions require code review and evaluation against the eval suite before
  deployment (`evals/run_evals.py` must show kappa ≥ 0.60 and all-guardrails ≥ 85%)

### 3.2 Change classification

| Change type | Classification | Required review |
|-------------|----------------|-----------------|
| Spelling/grammar fix, no semantic change | Non-material | Single reviewer |
| Output format adjustment | Non-material | Developer + validator |
| Adding/removing tool instructions | Material | Full eval re-run + sign-off |
| Risk threshold wording changes | Material | Legal + Compliance + MRM |
| Adding new tool references | Material | Full eval re-run + sign-off |

### 3.3 Prompt injection risk

The agent's tools return data from a controlled internal database (validated parquets
from the IEEE-CIS pipeline). External data sources (merchant databases, web lookups)
are not implemented. This architectural decision eliminates the primary vector for
prompt injection — malicious content in tool outputs that redirects agent behaviour.

If external data sources are added in future releases, each must be sanitised before
injection into the LLM context (strip special tokens, length-limit, format-validate).

---

## 4. Guardrail Framework

Four automated checks are applied to every agent output before it reaches an analyst:

| Check | What it catches | Pass criteria |
|-------|-----------------|---------------|
| `validate_output_schema` | Missing fields, wrong types, invalid enum values | All required keys present, types correct |
| `check_shap_consistency` | Agent citing SHAP features not actually in top SHAP contributors | No feature names in SHAP claims that don't appear in tool output |
| `check_no_pii_leakage` | Card numbers, SSNs, email addresses, account numbers in output | Zero pattern matches across all output fields |
| `check_action_consistency` | Business rule violations (e.g. auto_decline without reason codes) | No hard violations; warnings logged separately |

**Guardrail failure handling:**
- If any guardrail fails, the output is flagged for mandatory human review before
  any action is taken
- Outputs failing schema validation are rejected entirely (not presented to analysts)
- Outputs failing SHAP consistency or PII checks are presented with a prominent
  warning banner in the UI

---

## 5. Evaluation Methodology

The eval suite (`evals/run_evals.py`) is the primary tool for validating agent
behaviour before deployment and after any prompt or tool change.

### 5.1 Eval set design

30 hand-curated cases across 4 scenario types:

| Type | N | Description | Expected action |
|------|---|-------------|-----------------|
| Clear fraud | 10 | Score >0.95, confirmed fraud | auto_decline |
| Clear legitimate | 8 | Score <0.05, confirmed legit | auto_approve |
| Ambiguous middle | 7 | Score 0.40–0.70, mixed outcomes | escalate |
| Adversarial | 5 | Edge cases designed to surface failure modes | varies |

Adversarial cases specifically include:
- Large amount + low model score (tests over-reliance on amount)
- High score + confirmed legitimate (false positive — tests deference vs. dogmatism)
- Low score + confirmed fraud (missed fraud — tests signal extraction beyond model score)
- Overnight timing + elevated amount but clean (tests spurious anomaly detection)
- Borderline score + actual fraud (tests confidence calibration)

### 5.2 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Schema pass rate | % outputs passing schema validation | ≥ 95% |
| SHAP consistency pass rate | % outputs with grounded feature citations | ≥ 90% |
| PII pass rate | % outputs with no PII leakage | 100% |
| Action consistency pass rate | % outputs with no business rule violations | ≥ 98% |
| All-guardrails pass rate | % outputs passing all 4 checks | ≥ 88% |
| Action exact match | % predicted actions matching expected label | ≥ 70% |
| Cohen's kappa | Inter-rater agreement on 3-class action | ≥ 0.60 |
| Priority score gap | Mean(fraud priority) − Mean(legit priority) | > 2.0 |
| Adversarial accuracy | % adversarial cases handled correctly | ≥ 60% |

### 5.3 Limitations of the eval set

1. **Ground truth is the expected label, not analyst consensus.** A single human
   (the developer) labelled each case. Production validation would require at least
   two independent labellers with measured inter-rater reliability.

2. **30 cases is illustrative.** For a production deployment, a minimum of 200 cases
   covering the full operational score distribution is recommended.

3. **Non-determinism.** The LLM produces different outputs across runs. Kappa and
   pass rates should be averaged across 3+ independent runs before treating them as
   stable estimates.

4. **No regression testing against live data.** The eval set uses historical validation
   data. Novel fraud patterns not in the training distribution may produce qualitatively
   different agent behaviour.

---

## 6. Third-Party Model Risk (Anthropic Claude)

Using a third-party LLM introduces vendor risk that requires separate management:

| Risk | Mitigation |
|------|-----------|
| Model version changes affecting behaviour | Pin to specific model version; re-evaluate on upgrade |
| API availability / latency | Agent is advisory-only; fraud decision pipeline continues without it |
| Data confidentiality | No raw customer PII sent to API; only encoded features and SHAP values |
| Vendor financial/operational failure | Guardrail layer and underlying ML model operate independently |

**Data minimisation:** The agent tools transmit only the minimum data necessary —
transaction amounts (no card numbers), encoded categorical values (no raw strings
for sensitive fields), and SHAP values. Full card numbers, SSNs, and account
identifiers are never included in API calls.

---

## 7. Human-in-the-Loop Requirements

The agent system is designed around the principle of **human-in-the-loop for all
consequential decisions.** This is not just a best practice — it is a regulatory
requirement under ECOA and the CFPB's AI guidance.

**Mandatory human review triggers:**
- Any `recommended_action == "auto_decline"` (adverse action requires human sign-off)
- Any output that fails one or more guardrails
- Any case where `priority_score ≥ 9` (highest-risk tier)
- Any case where the agent notes an unexplained tension in the evidence

**Prohibited automation:**
- The agent's `recommended_action` field must not be wired directly to a transaction
  approval/decline system without human intervention
- SAR (Suspicious Activity Report) filings must not be automated based solely on
  agent output; human review and verification is legally required

---

## 8. Ongoing Monitoring

| Activity | Frequency | Owner | Threshold |
|----------|-----------|-------|-----------|
| Guardrail pass rates on production outputs | Weekly | ML Ops | All-guardrails < 88% → alert |
| Prompt version review | Quarterly | MRM team | After any trigger or scheduled |
| Eval suite re-run | On prompt change + quarterly | ML Developer | Kappa < 0.60 → block deployment |
| Anthropic model version audit | On each Anthropic release | ML Developer | Review release notes |
| PII leakage audit (sample) | Monthly | Privacy Officer | Any positive → immediate investigation |

---

## 9. Known Limitations

1. **Context window constraints.** Complex transactions with long customer histories
   may exceed the token budget. Tools are designed to return concise summaries, but
   extreme cases may require truncation.

2. **V-feature opacity.** The agent reasons about SHAP-important Vesta V-features
   but cannot explain their real-world meaning (Vesta has not disclosed this). Agent
   narratives for V-feature-driven cases are inherently less interpretable.

3. **Training data cutoff.** The underlying Claude model has a knowledge cutoff
   that may not reflect emerging fraud patterns (e.g. new synthetic identity
   methodologies). The fraud ML model's SHAP explanations partially compensate by
   grounding agent reasoning in current data, but the LLM's general reasoning
   patterns are fixed at training time.

4. **Eval set size.** 30 cases provides directional signal, not statistical confidence.
   Widen to 200+ before production deployment.
