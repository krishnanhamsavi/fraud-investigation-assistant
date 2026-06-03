"""Agent evaluation harness — runs all eval cases, applies guardrails, reports metrics.

Caches individual agent results so re-runs are cheap (only new/failed cases are re-run).

Usage:
    uv run python evals/run_evals.py                  # run all, use cache
    uv run python evals/run_evals.py --force          # re-run everything
    uv run python evals/run_evals.py --report-only    # print metrics from cache
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from dotenv import load_dotenv
load_dotenv()

from src.agent.guardrails import run_all_guardrails
from src.agent.investigator import investigate

EVAL_SET_PATH = Path("evals/agent_eval_set.jsonl")
CACHE_DIR = Path("evals/results")
RESULTS_PATH = CACHE_DIR / "eval_summary.json"

SEP = "=" * 64
ACTION_MAP = {"auto_approve": 0, "escalate": 1, "auto_decline": 2}
ACTION_LABELS = {0: "auto_approve", 1: "escalate", 2: "auto_decline"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SET_PATH} not found. "
            "Run: uv run python scripts/generate_eval_set.py"
        )
    cases = []
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def cache_path(eval_id: str) -> Path:
    return CACHE_DIR / f"{eval_id}.json"


def load_cached(eval_id: str) -> dict | None:
    p = cache_path(eval_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_cache(eval_id: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(eval_id).write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cohen_kappa(y_true: list[int], y_pred: list[int], n_classes: int = 3) -> float:
    """Compute Cohen's kappa without sklearn dependency."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Observed agreement
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    # Expected agreement
    pe = sum(
        (y_true.count(k) / n) * (y_pred.count(k) / n)
        for k in range(n_classes)
    )
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def compute_metrics(cases: list[dict], results: list[dict]) -> dict:
    """Compute all eval metrics from cases + cached results."""
    n = len(results)
    if n == 0:
        return {}

    # Guardrail pass rates
    schema_pass = sum(r.get("guardrails", {}).get("schema_validation", {}).get("passed", False)
                      for r in results)
    shap_pass = sum(r.get("guardrails", {}).get("shap_consistency", {}).get("passed", False)
                    for r in results)
    pii_pass = sum(r.get("guardrails", {}).get("pii_leakage", {}).get("passed", False)
                   for r in results)
    action_guard_pass = sum(r.get("guardrails", {}).get("action_consistency", {}).get("passed", False)
                            for r in results)
    all_pass = sum(
        all(v.get("passed", False) for v in r.get("guardrails", {}).values())
        for r in results
    )

    # Action agreement
    expected = [ACTION_MAP.get(c["expected_action"], 1) for c in cases]
    predicted = []
    for r in results:
        action = r.get("output", {}).get("recommended_action", "escalate")
        predicted.append(ACTION_MAP.get(action, 1))

    exact_match = sum(a == b for a, b in zip(expected, predicted))
    kappa = cohen_kappa(expected, predicted)

    # Priority score by actual fraud label
    fraud_priorities = [
        r.get("output", {}).get("priority_score", 5)
        for r, c in zip(results, cases) if c["actual_fraud"] == 1
    ]
    legit_priorities = [
        r.get("output", {}).get("priority_score", 5)
        for r, c in zip(results, cases) if c["actual_fraud"] == 0
    ]

    # Confusion breakdown
    confusion: dict[str, dict[str, int]] = {
        "auto_approve": {"auto_approve": 0, "escalate": 0, "auto_decline": 0},
        "escalate":     {"auto_approve": 0, "escalate": 0, "auto_decline": 0},
        "auto_decline": {"auto_approve": 0, "escalate": 0, "auto_decline": 0},
    }
    for e, p in zip(expected, predicted):
        exp_label = ACTION_LABELS[e]
        pred_label = ACTION_LABELS[p]
        confusion[exp_label][pred_label] += 1

    # Adversarial case performance
    adv_indices = [i for i, c in enumerate(cases) if c.get("adversarial")]
    adv_correct = sum(
        expected[i] == predicted[i] for i in adv_indices
    )

    return {
        "n_cases": n,
        "n_completed": sum(r.get("error") is None for r in results),
        "n_errors": sum(r.get("error") is not None for r in results),
        "guardrail_pass_rates": {
            "schema_validation": round(schema_pass / n, 3),
            "shap_consistency": round(shap_pass / n, 3),
            "pii_leakage": round(pii_pass / n, 3),
            "action_consistency": round(action_guard_pass / n, 3),
            "all_guardrails": round(all_pass / n, 3),
        },
        "action_agreement": {
            "exact_match_rate": round(exact_match / n, 3),
            "cohen_kappa": round(kappa, 3),
            "n_cases": n,
        },
        "priority_score_by_label": {
            "mean_fraud_priority": round(float(np.mean(fraud_priorities)), 2) if fraud_priorities else None,
            "mean_legit_priority": round(float(np.mean(legit_priorities)), 2) if legit_priorities else None,
            "n_fraud": len(fraud_priorities),
            "n_legit": len(legit_priorities),
        },
        "adversarial_performance": {
            "n_adversarial": len(adv_indices),
            "n_correct": adv_correct,
            "accuracy": round(adv_correct / len(adv_indices), 3) if adv_indices else None,
        },
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(case: dict, api_key: str) -> dict:
    """Run a single eval case and return a cacheable result dict."""
    result = investigate(transaction_id=case["transaction_id"], api_key=api_key)
    return {
        "eval_id": case["eval_id"],
        "transaction_id": case["transaction_id"],
        "output": result.raw_output,
        "guardrails": result.guardrail_results,
        "tool_call_log": result.tool_call_log,
        "iterations_used": result.iterations_used,
        "error": result.error,
    }


def print_metrics(metrics: dict, cases: list[dict], results: list[dict]) -> None:
    print(f"\n{SEP}")
    print("EVAL RESULTS")
    print(SEP)

    print(f"\n  Cases completed : {metrics['n_completed']} / {metrics['n_cases']}")
    if metrics["n_errors"]:
        print(f"  Errors          : {metrics['n_errors']} (see cache for details)")

    print(f"\n  Guardrail pass rates:")
    for check, rate in metrics["guardrail_pass_rates"].items():
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"    {check:<22}  {bar}  {rate:.0%}")

    m = metrics["action_agreement"]
    print(f"\n  Action agreement:")
    print(f"    Exact match rate : {m['exact_match_rate']:.0%}  ({int(m['exact_match_rate']*m['n_cases'])}/{m['n_cases']})")
    print(f"    Cohen's kappa    : {m['cohen_kappa']:.3f}  ", end="")
    k = m["cohen_kappa"]
    print("(substantial)" if k >= 0.6 else ("moderate)" if k >= 0.4 else "(fair)"))

    p = metrics["priority_score_by_label"]
    print(f"\n  Priority score (1–10) by ground truth:")
    print(f"    Actual fraud     : {p['mean_fraud_priority']:.2f}  (n={p['n_fraud']})")
    print(f"    Actual legit     : {p['mean_legit_priority']:.2f}  (n={p['n_legit']})")
    gap = (p["mean_fraud_priority"] or 0) - (p["mean_legit_priority"] or 0)
    print(f"    Gap (fraud–legit): {gap:+.2f}  {'✓ fraud prioritised higher' if gap > 0 else '✗ expected fraud > legit'}")

    adv = metrics["adversarial_performance"]
    print(f"\n  Adversarial cases: {adv['n_correct']}/{adv['n_adversarial']} correct  ({adv['accuracy']:.0%})")

    print(f"\n  Action confusion matrix (rows=expected, cols=predicted):")
    cols = ["auto_approve", "escalate", "auto_decline"]
    col_header = "Expected / Predicted"
    header = f"    {col_header:<20}" + "".join(f"  {c:<14}" for c in cols)
    print(header)
    for exp, row_d in metrics["confusion_matrix"].items():
        row_str = f"    {exp:<20}" + "".join(f"  {row_d[c]:<14}" for c in cols)
        print(row_str)

    print(f"\n{SEP}")


def main(force: bool = False, report_only: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_eval_set()
    print(f"{SEP}\nAGENT EVAL SUITE — {len(cases)} cases\n{SEP}")

    if not report_only:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set. Create a .env file.")
            sys.exit(1)

        # Pre-warm context once (avoids reload per case)
        from src.agent.tools import get_context
        print("Loading model context (one-time) …")
        get_context()

        for i, case in enumerate(cases, 1):
            cached = None if force else load_cached(case["eval_id"])
            if cached is not None:
                print(f"  [{i:02d}/{len(cases)}] {case['eval_id']}  (cached)")
                continue

            print(f"  [{i:02d}/{len(cases)}] {case['eval_id']}  {case['scenario_type']}"
                  f"  txn={case['transaction_id']}  score={case['model_score']:.3f} …", end="", flush=True)
            try:
                result = run_case(case, api_key)
                save_cache(case["eval_id"], result)
                action = result.get("output", {}).get("recommended_action", "?")
                expected = case["expected_action"]
                match = "✓" if action == expected else "✗"
                print(f"  {match}  predicted={action}  expected={expected}")
            except Exception as exc:
                save_cache(case["eval_id"], {"eval_id": case["eval_id"], "error": str(exc), "output": {}, "guardrails": {}, "tool_call_log": []})
                print(f"  ERROR: {exc}")

    # Load all cached results
    results = []
    for case in cases:
        cached = load_cached(case["eval_id"])
        if cached:
            results.append(cached)

    if not results:
        print("No results found. Run without --report-only first.")
        return

    metrics = compute_metrics(cases[:len(results)], results)
    print_metrics(metrics, cases, results)

    RESULTS_PATH.write_text(json.dumps({"metrics": metrics, "n_cases": len(cases)}, indent=2))
    print(f"Full results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run all cases (ignore cache)")
    parser.add_argument("--report-only", action="store_true", help="Print metrics from cache, no API calls")
    args = parser.parse_args()
    main(force=args.force, report_only=args.report_only)
