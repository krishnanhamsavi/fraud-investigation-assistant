# Fraud Investigation Assistant

**An AI-assisted decision support system for fraud analysts — built with the governance layer most GenAI demos skip.**

[**Live Demo**](https://fraud-investigation-assistant-j4fpvpfna4yyh94ujxkccg.streamlit.app/) · [Key Results](#key-results) · [Architecture](#architecture) · [Governance & Documentation](#governance--documentation)

---

## The Problem

Fraud models are hard to act on: a score of 0.87 tells an analyst nothing about *why* a transaction looks suspicious. Large language models can reason about evidence and write clear narratives — but they hallucinate, and in a regulated risk environment an ungrounded explanation is worse than none.

This project explores how to combine the two safely:

> **Precise model signals + grounded AI reasoning + automated safety checks + a human making the final call.**

## How It Works

```
Transaction ──► XGBoost scores it ──► SHAP explains it ──► Claude investigates it
                                                                    │
                                              4 guardrails validate the output
                                                                    │
                                              Analyst reviews and decides
```

1. **Model flags** — XGBoost predicts fraud probability on IEEE-CIS data (590k transactions)
2. **SHAP explains** — Per-transaction feature attributions, mapped to 15 ECOA-style reason codes (R01–R15)
3. **AI investigates** — A Claude agent pulls transaction details, SHAP evidence, and customer history through a tool-use loop, then produces a structured investigation (risk level, recommended action, narrative)
4. **Guardrails validate** — Schema validation, SHAP consistency, PII leakage detection, action logic checks
5. **Analyst decides** — The system recommends; it never auto-declines. Human-in-the-loop by design.

## Key Results

| Metric | Value | Why it matters |
| --- | --- | --- |
| ROC-AUC | 0.9126 | Strong fraud/legit separation |
| FDR @ 3% review rate | 47.8% | Catches nearly half of fraud while reviewing only 3% of transactions |
| Guardrail pass rate | 97% | AI output is structurally valid, grounded, and PII-safe |
| Cohen's kappa vs. labels | 0.851 | Substantial agreement between agent recommendations and curated ground truth |
| Score PSI | 0.00093 | Stable score distribution across the monitoring window |

Evaluated against a 30-case suite spanning fraud, legitimate, ambiguous, and adversarial scenarios. Full methodology in [`evals/`](evals/) and [`docs/`](docs/).

## Architecture

The project is organized as a seven-phase pipeline, mirroring how a model risk team would actually stand up a system like this:

| Phase | What it does | Highlights |
| --- | --- | --- |
| 1. Data pipeline | Load, engineer features, split | Time-based train/val split to prevent temporal leakage (472k / 118k) |
| 2. Champion model | XGBoost + SHAP | `scale_pos_weight` for 3.5% class imbalance, TreeExplainer attributions |
| 3. Challenger benchmarking | Compare alternatives | Logistic Regression (0.8375 AUC) and LightGBM (0.8306) vs. champion |
| 4. Monitoring & fairness | Drift and disparate impact | PSI, KS-tests, 4/5ths-rule audit on proxy variables |
| 5. LLM agent | Claude tool-use loop | Structured JSON output, grounded in model evidence, 6-iteration cap |
| 6. Eval suite | Test the agent like a model | 30 curated cases, 4 automated guardrails, adversarial inputs |
| 7. Streamlit UI | Analyst-facing dashboard | Transaction picker, SHAP panel, AI investigation, What-If mode |

## Governance & Documentation

The differentiating layer of this project. SR 11-7 model risk principles are extended to cover the GenAI component:

- [**Model Documentation**](docs/01_model_documentation.md) — architecture, hyperparameters, validation approach
- [**Challenger Analysis**](docs/02_challenger_analysis.md) — champion selection rationale
- [**Fairness Analysis**](docs/03_fairness_analysis.md) — disparate impact audit, proxy variable discussion
- [**Monitoring Plan**](docs/04_monitoring_plan.md) — PSI thresholds, retraining triggers
- [**LLM Governance**](docs/05_llm_governance.md) — guardrail framework, applying SR 11-7 to GenAI
- [**Reason Codes**](docs/07_reason_codes.md) — ECOA R01–R15 mappings from SHAP features

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/krishnanhamsavi/fraud-investigation-assistant.git
cd fraud-investigation-assistant
pip install -r requirements.txt

# 2. Configure
echo "ANTHROPIC_API_KEY=sk-..." >> .env

# 3. Add data
# Download train_transaction.csv and train_identity.csv from the
# IEEE-CIS Kaggle competition and place them in data/raw/

# 4. Launch the app
streamlit run src/app/streamlit_app.py
```

To reproduce the full pipeline (training, benchmarking, monitoring, evals), run the phase scripts in order:

```bash
python scripts/phase1_stats.py
python -m src.models.train && python scripts/phase2_shap.py
python scripts/phase3_compare.py
python scripts/phase4_monitoring.py
python scripts/phase5_demo.py
python scripts/generate_eval_set.py && python evals/run_evals.py
```

## Project Structure

```
├── src/
│   ├── data/        # Loading & preprocessing
│   ├── models/      # XGBoost champion + challengers
│   ├── explain/     # SHAP explanations & reason codes
│   ├── agent/       # LLM agent, tools, guardrails, prompts
│   └── app/         # Streamlit UI
├── scripts/         # Pipeline scripts (phases 1–6)
├── evals/           # Agent eval suite
├── docs/            # Governance documentation
├── notebooks/       # EDA & analysis
└── tests/           # Unit tests
```

## Technologies

**ML:** XGBoost, scikit-learn, pandas · **Explainability:** SHAP · **GenAI:** Anthropic Claude (tool use) · **UI:** Streamlit · **Evaluation:** Cohen's kappa, PSI, disparate impact ratio

## Scope & Limitations

This is a demonstration system built on public Kaggle data, designed to show what a *governed* GenAI deployment in fraud operations would look like — not a system that has processed live traffic. The eval suite is hand-curated (30 cases) and would need substantial expansion, blind labeling, and inter-rater reliability checks before any real-world use. Fairness analysis relies on proxy variables since the dataset contains no protected-class attributes.

Stating these limits is part of the point: knowing what a model *can't* claim is as important as what it can.

## Author

**Hamsavahini Krishnan** — Data analyst working at the intersection of risk, analytics, and AI governance.

[LinkedIn](https://www.linkedin.com/in/krishnanhamsavi) · [GitHub](https://github.com/krishnanhamsavi)

Questions or feedback? Open an issue or reach out.
