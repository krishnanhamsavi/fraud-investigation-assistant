# Fraud Investigation Assistant — AI-Powered Decision Support System

A production-grade fraud detection system combining **XGBoost machine learning, SHAP explanations, and Claude AI reasoning** with automated safety guardrails. Built to demonstrate SR 11-7 compliance, GenAI governance, and human-in-the-loop fraud analytics.

**[Try the Live Demo](#deployment)** | **[Key Results](#results)** | **[Architecture](#architecture)**

---

## Overview

This system investigates suspicious transactions by:
1. **Model flags** — XGBoost predicts fraud probability (ROC-AUC: 0.9126)
2. **SHAP explains** — Which transaction features drove the score
3. **AI reasons** — Claude synthesizes evidence into a structured investigation
4. **Guardrails validate** — 4 automated checks ensure no hallucinations or PII leaks
5. **Analyst decides** — Humans make final approval/decline call

**Why this matters:** Traditional fraud models are black boxes. GenAI can reason about evidence but hallucinates. This system combines both: precise model signals + AI reasoning + automated safety checks.

---

## Key Results

| Metric | Value | What it means |
|--------|-------|---------------|
| **ROC-AUC** | 0.9126 | Model separates fraud/legit well |
| **PR-AUC** | 0.5291 | Handles class imbalance (3.5% fraud) |
| **FDR @ 3%** | 47.83% | Catches ~48% of fraud at 3% review rate |
| **Guardrail pass rate** | 97% | AI output is valid & safe |
| **Cohen's kappa** | 0.851 | Substantial agreement on actions |
| **Priority gap** | +4.67 | Fraud prioritized 4.67pts higher than legit |

---

## Architecture: 7 Phases

### Phase 0: Scaffolding
- Project structure, environment, data paths

### Phase 1: Data Pipeline
- Load IEEE-CIS fraud dataset (590k transactions)
- Feature engineering (log amounts, time-of-day, composite features)
- Time-based train/val split (no temporal leakage)
- OrdinalEncoder for categoricals
- **Output:** 472k train, 118k val samples

### Phase 2: Champion Model + SHAP
- XGBoost with `scale_pos_weight` for imbalance
- Early stopping on validation AUC
- SHAP TreeExplainer for per-transaction explanations
- 15 ECOA reason codes (R01–R15) mapped from SHAP features
- **Output:** ROC-AUC 0.9126, PR-AUC 0.5291

### Phase 3: Challenger Benchmarking
- Logistic Regression (baseline): 0.8375 AUC
- LightGBM (high cardinality): 0.8306 AUC
- Champion selected on accuracy + latency trade-off

### Phase 4: Monitoring & Fairness
- Population Stability Index (PSI) for score drift
- KS-test for distribution shifts
- Disparate impact analysis using 4/5ths rule
- Proxy variables (ProductCD, card type) for fairness audit
- **Output:** Score PSI 0.00093 (stable), no severe DI violations

### Phase 5: LLM Agent (Claude)
- Tool-use loop: transaction details → SHAP → customer history → reason codes
- Structured JSON output with risk/action/narrative
- 6-iteration max with proper JSON extraction
- **Output:** Agent narratives grounded in tools, no hallucination

### Phase 6: Agent Eval Suite
- 30 hand-curated cases (fraud/legit/ambiguous/adversarial)
- 4 guardrails: schema validation, SHAP consistency, PII leakage, action logic
- Metrics: 90% exact match, 0.851 kappa, 97% all-guardrails pass

### Phase 7: Streamlit UI
- Professional dashboard with dark sidebar, warm color palette
- Transaction picker with advanced search/filtering
- Left panel: details + SHAP + customer history
- Right panel: AI investigation + guardrail results
- Plain English summaries + What If mode (estimate impact of changes)
- User-friendly guidance throughout

---

## Live Demo

**[fraud-investigation-assistant.streamlit.app](https://fraud-investigation-assistant-j4fpvpfna4yyh94ujxkccg.streamlit.app)**

The hosted demo runs in a self-contained mode: it renders 30 real investigation
cases from pre-computed eval results (transaction details, SHAP explanations,
the agent's narrative, and guardrail outcomes). The full pipeline — model
training, live SHAP, and live agent runs — executes locally (see Quick Start).

---

## Quick Start

### Local Setup

1. **Clone & install**
   ```bash
   git clone https://github.com/krishnanhamsavi/fraud-investigation-assistant.git
   cd fraud-investigation-assistant
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=sk-..." >> .env
   ```

3. **Download data** (from IEEE-CIS Kaggle competition)
   ```bash
   # Place train_transaction.csv and train_identity.csv in data/raw/
   # Or update src/data/load.py with your data path
   ```

4. **Run the pipeline (optional)**
   ```bash
   # Phase 1: Data preprocessing
   python scripts/phase1_stats.py
   
   # Phase 2: Train champion model + SHAP
   python -m src.models.train
   python scripts/phase2_shap.py
   
   # Phase 3: Compare challengers
   python scripts/phase3_compare.py
   
   # Phase 4: Monitoring & fairness
   python scripts/phase4_monitoring.py
   
   # Phase 5: LLM agent demo
   python scripts/phase5_demo.py
   
   # Phase 6: Agent eval suite
   python scripts/generate_eval_set.py
   python evals/run_evals.py
   ```

5. **Launch the app**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```
   Opens at `http://localhost:8501`

---

## Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Phase 7: Production Streamlit UI"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repo
   - Set environment variable: `ANTHROPIC_API_KEY`
   - Deploy

3. **Share the link**
   - URL: `https://share.streamlit.io/your-user/fraud-agent-mrm/main/src/app/streamlit_app.py`

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app/streamlit_app.py"]
```

---

## Project Structure

```
fraud-agent-mrm/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # XGBoost champion + challengers
│   ├── explain/        # SHAP explanations & reason codes
│   ├── agent/          # LLM agent, tools, guardrails, prompts
│   └── app/            # Streamlit UI
├── scripts/            # Pipeline scripts (phase1–6)
├── evals/              # Agent eval suite
├── notebooks/          # EDA, analysis
├── data/
│   ├── raw/            # Original CSVs
│   └── processed/      # Parquets, encoders
├── docs/               # Model documentation, fairness analysis, monitoring plan
├── tests/              # Unit tests
├── requirements.txt    # Dependencies
├── .env.example        # Template for environment variables
└── README.md           # This file
```

---

## Documentation

- **[Model Documentation](docs/01_model_documentation.md)** — Champion architecture, hyperparameters, validation approach
- **[Challenger Analysis](docs/02_challenger_analysis.md)** — Why Champion was selected
- **[Fairness Analysis](docs/03_fairness_analysis.md)** — Disparate impact audit, proxy variables
- **[Monitoring Plan](docs/04_monitoring_plan.md)** — PSI thresholds, retraining triggers
- **[LLM Governance](docs/05_llm_governance.md)** — SR 11-7 extended to GenAI, guardrails framework
- **[SR 11-7 Overview](docs/06_sr_11_7_overview.md)** — model risk management framework applied to this project
- **[Reason Codes](docs/07_reason_codes.md)** — ECOA R01–R15 mappings

---

## Technologies

- **ML:** XGBoost, scikit-learn, pandas, numpy
- **Explainability:** SHAP (TreeExplainer)
- **GenAI:** Anthropic Claude (tool-use architecture)
- **UI:** Streamlit with custom CSS
- **Evaluation:** Cohen's kappa, confusion matrix, disparate impact ratio, PSI

---

## For Portfolio / Interview

**One-line pitch:** An end-to-end fraud investigation system that pairs an
explainable ML model with a governed AI agent — designed the way a bank's
model-risk function would actually require: auditable, fair, monitored, and
human-in-the-loop.

**What it demonstrates:**
- Full ML system ownership (data → model → explanation → monitoring → UI → deploy)
- Financial-services governance fluency (SR 11-7, ECOA reason codes, fairness, drift)
- Responsible GenAI integration (tool-use agent + guardrail layer + measured eval suite)
- Communication to multiple audiences (plain-English, SHAP, regulatory docs)

> Note: this is a portfolio demonstration on a public dataset — illustrative
> governance and a 30-case eval, not a production deployment.

---

## Author

**Hamsavi Krishnan** — [GitHub](https://github.com/krishnanhamsavi)

Built to demonstrate full-stack ML engineering, GenAI governance, and
model-risk-management practices for Fraud Analytics / MRM roles.
