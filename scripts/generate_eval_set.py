"""Generate the agent eval set from the validation split.

Selects 30 real TransactionIDs across 4 scenario types and writes
evals/agent_eval_set.jsonl. Run once before evals/run_evals.py.

Usage:
    uv run python scripts/generate_eval_set.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.models.champion import load_champion
from src.models.train import FEATURE_EXCLUDE

EVAL_PATH = Path("evals/agent_eval_set.jsonl")
RNG = np.random.default_rng(42)

ACTION_FOR_SCORE = {
    "high": "auto_decline",
    "medium": "escalate",
    "low": "auto_approve",
}


def score_to_risk(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def build_case(
    row: pd.Series,
    eval_id: str,
    scenario_type: str,
    scenario_desc: str,
    expected_action: str,
    expected_risk: str,
    adversarial: bool,
    notes: str = "",
) -> dict:
    return {
        "eval_id": eval_id,
        "transaction_id": int(row["TransactionID"]),
        "scenario_type": scenario_type,
        "scenario_description": scenario_desc,
        "expected_action": expected_action,
        "expected_risk": expected_risk,
        "model_score": round(float(row["score"]), 5),
        "actual_fraud": int(row["isFraud"]),
        "adversarial": adversarial,
        "notes": notes,
    }


def main() -> None:
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and validation data …")
    val = pd.read_parquet("data/processed/val.parquet")
    feature_cols = [c for c in val.columns if c not in FEATURE_EXCLUDE]
    model = load_champion()
    val["score"] = model.predict_proba(val[feature_cols])[:, 1]
    val["isFraud"] = val["isFraud"].astype(int)

    cases: list[dict] = []
    counter = 1

    def nxt() -> str:
        nonlocal counter
        cid = f"eval_{counter:03d}"
        counter += 1
        return cid

    # -----------------------------------------------------------------------
    # 1. CLEAR FRAUD (10 cases) — score > 0.95, isFraud = 1
    #    Agent should call auto_decline with supporting SHAP evidence.
    # -----------------------------------------------------------------------
    pool = val[(val["isFraud"] == 1) & (val["score"] > 0.95)].copy()
    sample = pool.sample(n=min(10, len(pool)), random_state=42)
    for _, row in sample.iterrows():
        cases.append(build_case(
            row, nxt(),
            "clear_fraud",
            "High-confidence fraud: model score >0.95 with actual isFraud=1",
            "auto_decline", "high",
            adversarial=False,
            notes="Expect multiple corroborating SHAP factors. Reason codes required.",
        ))

    # -----------------------------------------------------------------------
    # 2. CLEAR LEGITIMATE (8 cases) — score < 0.05, isFraud = 0
    #    Agent should call auto_approve. No adverse reason codes expected.
    # -----------------------------------------------------------------------
    pool = val[(val["isFraud"] == 0) & (val["score"] < 0.05)].copy()
    sample = pool.sample(n=min(8, len(pool)), random_state=42)
    for _, row in sample.iterrows():
        cases.append(build_case(
            row, nxt(),
            "clear_legitimate",
            "Low-risk transaction: model score <0.05, confirmed legitimate",
            "auto_approve", "low",
            adversarial=False,
            notes="Agent should approve. Adverse reason codes must be empty.",
        ))

    # -----------------------------------------------------------------------
    # 3. AMBIGUOUS MIDDLE (7 cases) — score 0.40–0.70, mix of fraud/legit
    #    Agent should recommend escalation to human analyst.
    # -----------------------------------------------------------------------
    pool = val[(val["score"] >= 0.40) & (val["score"] <= 0.70)].copy()
    sample = pool.sample(n=min(7, len(pool)), random_state=42)
    for _, row in sample.iterrows():
        actual_label = "fraud" if row["isFraud"] == 1 else "legitimate"
        cases.append(build_case(
            row, nxt(),
            "ambiguous_middle",
            f"Borderline score (0.40–0.70); actual outcome: {actual_label}",
            "escalate", "medium",
            adversarial=False,
            notes="Human review required. Agent should surface tensions, not over-conclude.",
        ))

    # -----------------------------------------------------------------------
    # 4. ADVERSARIAL CASES (5 cases)
    # -----------------------------------------------------------------------

    # A1: Large amount, low model score, confirmed legitimate.
    #     Tests whether agent over-weights amount vs model signal.
    pool = val[
        (val["isFraud"] == 0) &
        (val["score"] < 0.15) &
        (val["TransactionAmt"] > 500)
    ].copy()
    if not pool.empty:
        row = pool.nlargest(1, "TransactionAmt").iloc[0]
        cases.append(build_case(
            row, nxt(),
            "adversarial_high_amount_clean",
            "Large transaction amount but low model score — confirmed legitimate",
            "auto_approve", "low",
            adversarial=True,
            notes=(
                "Agent must ground its assessment in the MODEL score and SHAP, "
                "not in the raw dollar amount alone. Correct answer is approve."
            ),
        ))

    # A2: Highest-scoring false positive — model very confident but actually legit.
    #     Tests agent nuance when score and ground truth conflict.
    pool = val[(val["isFraud"] == 0) & (val["score"] > 0.90)].copy()
    if not pool.empty:
        row = pool.nlargest(1, "score").iloc[0]
        cases.append(build_case(
            row, nxt(),
            "adversarial_high_score_fp",
            "Model score >0.90 but transaction is confirmed legitimate (false positive)",
            "escalate", "high",
            adversarial=True,
            notes=(
                "Agent cannot know ground truth. Expected: escalate with high risk "
                "based on model signal. This tests whether the agent appropriately "
                "defers to human review rather than auto-declining a legitimate txn."
            ),
        ))

    # A3: Low score, actual fraud — model missed it.
    #     Tests whether agent catches contextual signals the model underweighted.
    pool = val[(val["isFraud"] == 1) & (val["score"] < 0.30)].copy()
    if not pool.empty:
        row = pool.sample(1, random_state=7).iloc[0]
        cases.append(build_case(
            row, nxt(),
            "adversarial_low_score_fn",
            "Low model score (<0.30) but actually fraudulent — model miss",
            "escalate", "medium",
            adversarial=True,
            notes=(
                "Agent should note low model confidence but escalate if other signals "
                "(amount, timing, history) are anomalous. Correct answer is not approve."
            ),
        ))

    # A4: Overnight timing + above-average amount, but legitimate.
    #     Tests whether agent flags timing as suspicious when overall risk is low.
    pool = val[
        (val["isFraud"] == 0) &
        (val["score"] < 0.25) &
        (val["tx_hour"] <= 4) &
        (val["TransactionAmt"] > 200)
    ].copy()
    if not pool.empty:
        row = pool.sample(1, random_state=42).iloc[0]
        cases.append(build_case(
            row, nxt(),
            "adversarial_nighttime_legit",
            "Overnight transaction with elevated amount — confirmed legitimate",
            "escalate", "low",
            adversarial=True,
            notes=(
                "Low model score but timing anomaly present. "
                "Agent should note the tension and escalate rather than approve outright."
            ),
        ))

    # A5: Mid-range score, confirmed fraud, but not a slam-dunk.
    #     Tests agent's confidence calibration on borderline actual fraud.
    pool = val[
        (val["isFraud"] == 1) &
        (val["score"] >= 0.35) &
        (val["score"] <= 0.55)
    ].copy()
    if not pool.empty:
        row = pool.sample(1, random_state=99).iloc[0]
        cases.append(build_case(
            row, nxt(),
            "adversarial_mid_score_fraud",
            "Borderline model score (0.35–0.55) but confirmed fraudulent",
            "escalate", "medium",
            adversarial=True,
            notes=(
                "Model uncertainty is real — agent should not over-state confidence. "
                "Escalate with clear enumeration of what corroborating signals exist."
            ),
        ))

    # -----------------------------------------------------------------------
    # Write JSONL
    # -----------------------------------------------------------------------
    EVAL_PATH.write_text(
        "\n".join(json.dumps(c) for c in cases) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(cases)} eval cases → {EVAL_PATH}")

    # Summary
    by_type: dict[str, int] = {}
    for c in cases:
        by_type[c["scenario_type"]] = by_type.get(c["scenario_type"], 0) + 1
    print("\nBreakdown:")
    for t, n in sorted(by_type.items()):
        adv = sum(1 for c in cases if c["scenario_type"] == t and c["adversarial"])
        print(f"  {t:<35} {n:>2} cases{'  (adversarial)' if adv else ''}")

    score_dist = [c["model_score"] for c in cases]
    print(f"\nScore range: {min(score_dist):.4f} – {max(score_dist):.4f}")
    print(f"Actual fraud cases: {sum(c['actual_fraud'] for c in cases)} / {len(cases)}")


if __name__ == "__main__":
    main()
