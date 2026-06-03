"""Output validation and hallucination guards for LLM-generated investigation reports.

Every check returns a dict with:
  passed  : bool
  message : human-readable explanation
  details : any additional diagnostic data
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Required schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "transaction_id",
    "risk_assessment",
    "recommended_action",
    "priority_score",
    "sar_narrative",
    "adverse_action_reason_codes",
    "supporting_evidence",
}

VALID_RISK_VALUES = {"low", "medium", "high"}
VALID_ACTION_VALUES = {"auto_approve", "escalate", "auto_decline"}


# ---------------------------------------------------------------------------
# Check 1 — Output schema validation
# ---------------------------------------------------------------------------

def validate_output_schema(output: dict) -> dict:
    """Verify all required keys are present and values are the right type/enum."""
    issues = []

    missing = REQUIRED_KEYS - set(output.keys())
    if missing:
        issues.append(f"Missing keys: {sorted(missing)}")

    if "risk_assessment" in output and output["risk_assessment"] not in VALID_RISK_VALUES:
        issues.append(
            f"risk_assessment={output['risk_assessment']!r} not in {VALID_RISK_VALUES}"
        )

    if "recommended_action" in output and output["recommended_action"] not in VALID_ACTION_VALUES:
        issues.append(
            f"recommended_action={output['recommended_action']!r} not in {VALID_ACTION_VALUES}"
        )

    if "priority_score" in output:
        ps = output["priority_score"]
        if not isinstance(ps, int) or not (1 <= ps <= 10):
            issues.append(f"priority_score={ps!r} must be an integer 1–10")

    if "supporting_evidence" in output:
        ev = output["supporting_evidence"]
        if not isinstance(ev, list):
            issues.append("supporting_evidence must be a list")
        else:
            for i, item in enumerate(ev):
                for k in ("claim", "source", "value"):
                    if k not in item:
                        issues.append(f"supporting_evidence[{i}] missing key '{k}'")

    if "adverse_action_reason_codes" in output:
        if not isinstance(output["adverse_action_reason_codes"], list):
            issues.append("adverse_action_reason_codes must be a list")

    if "sar_narrative" in output:
        if not isinstance(output["sar_narrative"], str) or len(output["sar_narrative"]) < 20:
            issues.append("sar_narrative must be a non-trivial string (≥20 chars)")

    return {
        "passed": len(issues) == 0,
        "message": "Schema valid" if not issues else "; ".join(issues),
        "details": {"issues": issues},
    }


# ---------------------------------------------------------------------------
# Check 2 — SHAP consistency
# ---------------------------------------------------------------------------

def check_shap_consistency(output: dict, shap_data: dict | None) -> dict:
    """Verify the narrative/evidence references features that appear in top SHAP.

    If shap_data is None (tool wasn't called), the check is marked as a warning
    rather than a failure — the agent may have had a reason to skip it.
    """
    if shap_data is None:
        return {
            "passed": False,
            "message": "SHAP tool was not called — cannot verify grounding",
            "details": {"shap_called": False},
        }

    if "error" in shap_data:
        return {
            "passed": False,
            "message": f"SHAP tool returned error: {shap_data['error']}",
            "details": {"shap_error": shap_data["error"]},
        }

    top_features = {
        item["feature"].lower()
        for item in shap_data.get("top_shap_features", [])
    }

    if not top_features:
        return {
            "passed": False,
            "message": "No SHAP features returned — cannot verify grounding",
            "details": {},
        }

    # Check: does supporting_evidence cite at least one SHAP-grounded source?
    ev_sources = [e.get("source", "").lower() for e in output.get("supporting_evidence", [])]
    shap_sourced = any("shap" in s for s in ev_sources)

    # Check only SHAP-sourced evidence items + sar_narrative for feature-name hallucinations.
    # Restricting to SHAP-sourced claims avoids false positives when the agent mentions
    # a feature as a field value reference (e.g. "card1=10636" in customer history context)
    # rather than as a SHAP driver attribution.
    # Scope the hallucination check to SHAP-sourced evidence only.
    # The sar_narrative legitimately references any field as contextual detail
    # (e.g. "card1=10636 has zero history", "M1/M5/M6 are null") — checking it
    # produces false positives. Only SHAP-sourced claims make feature attribution
    # assertions that need grounding verification.
    shap_evidence_text = " ".join(
        e.get("claim", "") + " " + e.get("value", "")
        for e in output.get("supporting_evidence", [])
        if "shap" in e.get("source", "").lower()
    )
    check_text = shap_evidence_text.lower()

    features_cited = [f for f in top_features if f in check_text]

    # Pattern note: m[1-9] intentionally excludes "m0" — M0 is an encoded categorical
    # VALUE (missing flag), not a feature name. Feature names are M1–M9.
    cited_in_narrative = re.findall(
        r"\b(v\d+|c\d+|d\d+|m[1-9]\d*|card\d+)\b", check_text
    )
    hallucinated = [f for f in cited_in_narrative if f not in top_features]

    issues = []
    if not shap_sourced:
        issues.append("No supporting_evidence entries cite 'shap' as source")
    if hallucinated:
        issues.append(f"Narrative may cite features not in top SHAP: {hallucinated}")

    return {
        "passed": len(issues) == 0,
        "message": "SHAP grounding verified" if not issues else "; ".join(issues),
        "details": {
            "top_shap_features": sorted(top_features),
            "features_cited_in_narrative": features_cited,
            "potentially_hallucinated_features": hallucinated,
            "shap_sourced_in_evidence": shap_sourced,
        },
    }


# ---------------------------------------------------------------------------
# Check 3 — PII leakage
# ---------------------------------------------------------------------------

# Patterns for common PII in financial context
_PII_PATTERNS = [
    (re.compile(r"\b\d{13,16}\b"), "possible card number (13–16 digit sequence)"),
    (re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"), "possible SSN"),
    (re.compile(r"\b\d{3,4}\b(?=\s*(cvv|cvc|security code))", re.IGNORECASE), "possible CVV"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "email address"),
    (re.compile(r"\b\d{9,10}\b"), "possible account/routing number (9–10 digits)"),
]


def check_no_pii_leakage(output: dict) -> dict:
    """Scan all string fields in the output for patterns that look like PII."""
    # Collect all text from the output
    text_parts = []
    for key, val in output.items():
        if isinstance(val, str):
            text_parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_parts.extend(str(v) for v in item.values())

    full_text = " ".join(text_parts)
    findings = []

    for pattern, label in _PII_PATTERNS:
        matches = pattern.findall(full_text)
        if matches:
            # Redact matches for the log (don't reproduce PII in guardrail output)
            findings.append({
                "type": label,
                "count": len(matches),
                "redacted_example": "[REDACTED]",
            })

    return {
        "passed": len(findings) == 0,
        "message": "No PII patterns detected" if not findings else f"PII patterns found: {len(findings)} type(s)",
        "details": {"findings": findings},
    }


# ---------------------------------------------------------------------------
# Check 4 — Action consistency
# ---------------------------------------------------------------------------

def check_action_consistency(output: dict) -> dict:
    """Enforce business rules on recommended_action and reason codes.

    Rules:
    - auto_decline → adverse_action_reason_codes must be non-empty
    - auto_approve → adverse_action_reason_codes should be []
    - high risk + auto_approve → flag as inconsistent (warn only)
    """
    issues = []
    warnings = []

    action = output.get("recommended_action", "")
    codes = output.get("adverse_action_reason_codes", [])
    risk = output.get("risk_assessment", "")

    if action == "auto_decline" and not codes:
        issues.append(
            "auto_decline requires at least one adverse_action_reason_code (ECOA Reg B)"
        )

    if action == "auto_approve" and codes:
        warnings.append(
            f"auto_approve but {len(codes)} reason code(s) present — "
            "codes are only required for adverse actions"
        )

    if risk == "high" and action == "auto_approve":
        warnings.append(
            "high risk_assessment combined with auto_approve is contradictory — "
            "consider escalate instead"
        )

    if risk == "low" and action == "auto_decline":
        warnings.append(
            "low risk_assessment combined with auto_decline is contradictory"
        )

    return {
        "passed": len(issues) == 0,
        "message": (
            "Action consistency verified"
            if not issues and not warnings
            else "; ".join(issues + warnings)
        ),
        "details": {
            "hard_violations": issues,
            "warnings": warnings,
            "action": action,
            "risk": risk,
            "n_reason_codes": len(codes),
        },
    }


# ---------------------------------------------------------------------------
# Run all guardrails
# ---------------------------------------------------------------------------

def run_all_guardrails(output: dict, shap_data: dict | None = None) -> dict:
    """Execute all four guardrail checks. Returns a dict keyed by check name."""
    return {
        "schema_validation": validate_output_schema(output),
        "shap_consistency": check_shap_consistency(output, shap_data),
        "pii_leakage": check_no_pii_leakage(output),
        "action_consistency": check_action_consistency(output),
    }
