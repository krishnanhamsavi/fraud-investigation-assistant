"""Re-run guardrails against the saved phase5 demo JSON without re-calling the API.

Usage:
    uv run python scripts/phase5_recheck_guardrails.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.guardrails import run_all_guardrails

results_path = Path("docs/results/phase5_agent_demo.json")
if not results_path.exists():
    print("Run scripts/phase5_demo.py first.")
    sys.exit(1)

saved = json.loads(results_path.read_text())
SEP = "=" * 56

for entry in saved:
    print(SEP)
    print(f"Transaction {entry['transaction_id']} — {entry['scenario']}")
    print(SEP)

    output = entry.get("output", {})
    tool_log = entry.get("tool_call_log", [])

    shap_data = next(
        (log["output"] for log in tool_log if log["tool"] == "get_shap_explanation"),
        None,
    )

    guardrails = run_all_guardrails(output, shap_data)
    all_passed = True
    for check, res in guardrails.items():
        icon = "✓" if res["passed"] else "✗"
        print(f"  {icon} {check}: {res['message']}")
        if not res["passed"]:
            all_passed = False
    print(f"\n  Overall: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print()
