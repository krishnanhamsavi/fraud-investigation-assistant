"""Phase 5 demo — run the fraud investigation agent on 3 flagged transactions.

Requires ANTHROPIC_API_KEY in .env or environment.

Usage:
    uv run python scripts/phase5_demo.py
    uv run python scripts/phase5_demo.py --txn-ids 123456 234567 345678  # specific IDs
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from src.agent.investigator import investigate
from src.agent.tools import get_context
from src.models.train import FEATURE_EXCLUDE

RESULTS_DIR = Path("docs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 68


def pick_demo_transactions(ctx) -> list[int]:
    """Select 3 informative transaction IDs for the demo.

    1. True positive  — highest-scored actual fraud (clear case for agent)
    2. False positive — highest-scored non-fraud (tests agent nuance)
    3. Medium fraud   — moderate score actual fraud (borderline escalation)
    """
    val = ctx.val_encoded.copy()
    val["score"] = ctx.scores
    val["isFraud"] = val["isFraud"].astype(int)

    # 1. Top-scored fraud
    tp = val[val["isFraud"] == 1].nlargest(1, "score")
    txn1 = int(tp["TransactionID"].iloc[0])

    # 2. Top-scored non-fraud (false positive scenario)
    fp = val[val["isFraud"] == 0].nlargest(3, "score")
    # Prefer one with score > 0.85 to make it interesting
    high_fp = fp[fp["score"] > 0.85]
    txn2 = int(high_fp["TransactionID"].iloc[0] if not high_fp.empty else fp["TransactionID"].iloc[0])

    # 3. Medium-scored fraud (0.50 – 0.80)
    med = val[
        (val["isFraud"] == 1) &
        (val["score"] >= 0.50) &
        (val["score"] <= 0.80)
    ].sample(1, random_state=42)
    txn3 = int(med["TransactionID"].iloc[0])

    print(f"Selected transactions:")
    for label, tid in [("True positive (fraud)", txn1), ("False positive (legit)", txn2), ("Medium fraud", txn3)]:
        score = float(val[val["TransactionID"] == tid]["score"].iloc[0])
        actual = int(val[val["TransactionID"] == tid]["isFraud"].iloc[0])
        print(f"  {label:<28}  ID={tid:>10}  score={score:.4f}  actual_fraud={actual}")
    print()
    return [txn1, txn2, txn3]


def print_result(result, scenario_label: str) -> None:
    """Pretty-print one investigation result."""
    print(SEP)
    print(f"SCENARIO: {scenario_label}")
    print(SEP)

    if result.error:
        print(f"ERROR: {result.error}")
        return

    o = result.raw_output
    print(f"Transaction ID     : {o.get('transaction_id')}")
    print(f"Risk assessment    : {o.get('risk_assessment', '—').upper()}")
    print(f"Recommended action : {o.get('recommended_action', '—')}")
    print(f"Priority score     : {o.get('priority_score', '—')} / 10")
    print(f"Iterations used    : {result.iterations_used}")
    print()

    print("SAR Narrative:")
    print(f"  {o.get('sar_narrative', '[empty]')}")
    print()

    codes = o.get("adverse_action_reason_codes", [])
    if codes:
        print(f"Adverse Action Codes: {', '.join(codes)}")
        print()

    ev = o.get("supporting_evidence", [])
    if ev:
        print(f"Supporting Evidence ({len(ev)} items):")
        for i, item in enumerate(ev, 1):
            print(f"  [{i}] {item.get('claim', '')}")
            print(f"      Source: {item.get('source', '—')}  |  Value: {item.get('value', '—')}")
        print()

    print("Tool calls made:")
    for log in result.tool_call_log:
        print(f"  iter {log['iteration']}: {log['tool']}({log['input']})")
    print()

    print("Guardrail results:")
    all_passed = True
    for check, res in result.guardrail_results.items():
        icon = "✓" if res["passed"] else "✗"
        print(f"  {icon} {check}: {res['message']}")
        if not res["passed"]:
            all_passed = False
    print()
    print(f"Overall guardrail status: {'PASS' if all_passed else 'FAIL'}")


def main(txn_ids: list[int] | None = None) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Create a .env file with: ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    print(SEP)
    print("FRAUD INVESTIGATION AGENT — PHASE 5 DEMO")
    print(SEP)
    print("Loading model and data context …")
    ctx = get_context()

    if txn_ids:
        demo_ids = txn_ids
        labels = [f"Custom transaction {tid}" for tid in demo_ids]
    else:
        demo_ids = pick_demo_transactions(ctx)
        labels = [
            "True Positive — highest-scored fraud",
            "False Positive — highest-scored legitimate transaction",
            "Medium-scored fraud — borderline escalation",
        ]

    all_results = []
    for tid, label in zip(demo_ids, labels):
        print(f"\nInvestigating {tid} …")
        result = investigate(transaction_id=tid, api_key=api_key)
        print_result(result, label)
        all_results.append({
            "transaction_id": tid,
            "scenario": label,
            "output": result.raw_output,
            "guardrails": result.guardrail_results,
            "tool_call_log": result.tool_call_log,
            "iterations_used": result.iterations_used,
            "error": result.error,
        })

    # Save full results
    out_path = RESULTS_DIR / "phase5_agent_demo.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{SEP}")
    print(f"Full results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txn-ids", nargs="+", type=int, help="Specific TransactionIDs to investigate")
    args = parser.parse_args()
    main(txn_ids=args.txn_ids)
