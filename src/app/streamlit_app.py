"""Fraud Investigation System — Streamlit demo.

Run from project root:
    streamlit run src/app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Fraud Investigation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────── */
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }

[data-testid="stAppViewContainer"] { background: #f0f2f6; }

/* ── Sidebar ───────────────────────────── */
[data-testid="stSidebar"] { background: #0d1b2e; border-right: 1px solid #1e3050; }
[data-testid="stSidebar"] section { padding-top: 1rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #c9d6e8 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #1a2e4a; color: #e8edf5; border-color: #2d4a6e; }
[data-testid="stSidebarNav"] { display: none; }

/* Text input in sidebar */
[data-testid="stSidebar"] .stTextInput input {
    background: #1a2e4a !important; color: #e8edf5 !important;
    border: 1px solid #2d4a6e !important; border-radius: 6px;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: #6b82a3 !important; }

/* Multiselect: container + selected tags (fixes unreadable red boxes) */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1a2e4a !important; border-color: #2d4a6e !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563a8 !important; color: #ffffff !important;
    border-radius: 4px !important; font-size: 11px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] svg { fill: #cfe2ff !important; }
/* Selectbox dropdown popover (white menu → dark, readable text) */
[data-baseweb="popover"] [role="listbox"] { background: #ffffff !important; }
[data-baseweb="popover"] [role="option"] { color: #1a1a2e !important; font-weight: 500; }
[data-baseweb="popover"] [role="option"]:hover { background: #eaf2ff !important; color: #1a1a2e !important; }

/* Slider track + handle in sidebar */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #2563a8 !important; }

/* ── Cards ─────────────────────────────── */
.card {
    background: white;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 16px;
    border: 1px solid #e9ecef;
}
.card-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6c757d;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f0f2f5;
}

/* ── Risk badges ────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-high    { background:#fff0f0; color:#c0392b; border:1px solid #f5b7b1; }
.badge-medium  { background:#fef9e7; color:#b7770d; border:1px solid #f9e79f; }
.badge-low     { background:#eafaf1; color:#196f3d; border:1px solid #a9dfbf; }

/* ── Action display ─────────────────────── */
.action-block {
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.03em;
    display: inline-block;
    margin-top: 4px;
}
.action-approve  { background:#eafaf1; color:#196f3d; border:2px solid #27ae60; }
.action-escalate { background:#fef9e7; color:#b7770d; border:2px solid #f39c12; }
.action-decline  { background:#fff0f0; color:#c0392b; border:2px solid #e74c3c; }

/* ── Guardrail status ───────────────────── */
.guardrail-card {
    border-radius: 8px;
    padding: 14px 18px;
    border: 1px solid;
}
.guardrail-pass { background:#eafaf1; border-color:#a9dfbf; }
.guardrail-fail { background:#fff0f0; border-color:#f5b7b1; }
.guardrail-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
}
.guardrail-status {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.guardrail-pass .guardrail-status { color: #196f3d; }
.guardrail-fail .guardrail-status { color: #c0392b; }
.guardrail-pass .guardrail-label  { color: #239b56; }
.guardrail-fail .guardrail-label  { color: #c0392b; }
.guardrail-message { font-size: 12px; color: #555; margin-top: 3px; }

/* ── SAR narrative ──────────────────────── */
.sar-box {
    background: #f8f9fc;
    border-left: 3px solid #2980b9;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13.5px;
    line-height: 1.7;
    color: #2c3e50;
}

/* ── Detail table ───────────────────────── */
.detail-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #f5f6fa;
    font-size: 13.5px;
}
.detail-row:last-child { border-bottom: none; }
.detail-key   { color: #6c757d; font-weight: 500; }
.detail-value { color: #1a1a2e; font-weight: 600; text-align: right; }

/* ── Reason code pills ──────────────────── */
.code-pill {
    display: inline-block;
    background: #eaf2ff;
    color: #1a5276;
    border: 1px solid #aed6f1;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Score bar ──────────────────────────── */
.score-bar-bg {
    background: #e9ecef;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}

/* ── Misc ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #6c757d !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.stButton button {
    background: #1a2e4a; color: white; border: none;
    border-radius: 6px; font-weight: 600; font-size: 13px; padding: 8px 18px;
}
.stButton button:hover { background: #2d4a6e; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────
EVAL_SET_PATH = Path("evals/agent_eval_set.jsonl")
EVAL_CACHE_DIR = Path("evals/results")
RESULTS_DIR = Path("docs/results")
PROCESSED_DIR = Path("data/processed")

# Demo mode: when the full processed dataset and model are not present (e.g. on
# Streamlit Cloud, where the multi-hundred-MB parquets cannot be hosted), the app
# renders entirely from the pre-computed cached eval results. Every cached result
# embeds the full tool outputs (transaction details, SHAP, history, reason codes),
# so all 30 cases display fully without the model or raw data.
DEMO_MODE = not (PROCESSED_DIR / "val.parquet").exists()

RISK_COLORS = {"high": "#c0392b", "medium": "#b7770d", "low": "#196f3d"}
ACTION_LABELS = {
    "auto_approve": "Auto Approve",
    "escalate": "Escalate to Analyst",
    "auto_decline": "Auto Decline",
}


# ── Data loading (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and data…")
def load_context():
    from src.agent.tools import get_context
    return get_context()


@st.cache_data(show_spinner=False)
def fetch_txn_details(txn_id: int) -> dict:
    from src.agent.tools import get_transaction_details
    return get_transaction_details(txn_id)


@st.cache_data(show_spinner=False)
def fetch_shap(txn_id: int) -> dict:
    from src.agent.tools import get_shap_explanation
    return get_shap_explanation(txn_id, top_n=8)


@st.cache_data(show_spinner=False)
def fetch_history(card_id: int) -> dict:
    from src.agent.tools import get_customer_history
    return get_customer_history(card_id, lookback_days=30)


def load_eval_cases() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        return []
    cases = []
    with EVAL_SET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_cached_result(eval_id: str) -> dict | None:
    p = EVAL_CACHE_DIR / f"{eval_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def extract_tool_output(cached_result: dict | None, tool_name: str) -> dict:
    """Pull a tool's recorded output from a cached eval result's tool_call_log.

    Used in DEMO_MODE so the app can render transaction details, SHAP, history,
    and reason codes without loading the model or processed data.
    """
    if not cached_result:
        return {}
    for entry in cached_result.get("tool_call_log", []):
        if entry.get("tool") == tool_name:
            return entry.get("output") or {}
    return {}


def estimate_what_if_impact(base_txn: dict, modifications: dict) -> tuple[float, str]:
    """Estimate fraud score impact based on field changes (directional only).

    Returns: (estimated_new_score, impact_narrative)
    """
    original_score = base_txn.get("fraud_score", 0.5)
    original_amount = base_txn.get("TransactionAmt", 100)
    original_hour = int(base_txn.get("tx_hour", 12))

    delta = 0.0
    impacts = []

    # Amount impact (higher = more fraud risk)
    if "TransactionAmt" in modifications:
        new_amount = modifications["TransactionAmt"]
        amt_diff_pct = ((new_amount - original_amount) / original_amount) if original_amount > 0 else 0
        if amt_diff_pct > 0.2:
            delta += min(0.05, amt_diff_pct * 0.03)
            impacts.append(f"Higher amount → increased risk (+{delta*100:.1f}%)")
        elif amt_diff_pct < -0.2:
            delta -= min(0.03, abs(amt_diff_pct) * 0.02)
            impacts.append(f"Lower amount → decreased risk ({delta*100:.1f}%)")

    # Hour impact (late night = more fraud risk)
    if "tx_hour" in modifications:
        new_hour = int(modifications["tx_hour"])
        if new_hour < 6:  # Late night
            delta += 0.03
            impacts.append(f"Late night hours → slight increased risk (+3%)")
        elif new_hour > 18:  # Evening/night
            delta += 0.02
            impacts.append(f"Evening hours → slight increased risk (+2%)")
        else:  # Daytime
            delta -= 0.01
            impacts.append(f"Daytime hours → slight decreased risk (-1%)")

    estimated_score = max(0.0, min(1.0, original_score + delta))
    narrative = " | ".join(impacts) if impacts else "No significant impact detected."

    return estimated_score, narrative


# ── Plotting ───────────────────────────────────────────────────────────────

def shap_bar_chart(shap_features: list[dict]) -> plt.Figure:
    """Clean horizontal bar chart of SHAP contributions."""
    n = len(shap_features)
    features = [f["feature"] for f in shap_features]
    values = [f["shap_value"] for f in shap_features]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3.0, n * 0.52)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bars
    bars = ax.barh(range(n), values, color=colors, height=0.55, alpha=0.88, edgecolor="none")

    # Value labels
    for i, (val, bar) in enumerate(zip(values, bars)):
        x_pos = val + (0.005 if val >= 0 else -0.005)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"{val:+.4f}", va="center", ha=ha, fontsize=9,
                color=colors[i], fontweight="600")

    ax.set_yticks(range(n))
    ax.set_yticklabels(features, fontsize=10.5, color="#2c3e50")
    ax.invert_yaxis()
    ax.axvline(0, color="#adb5bd", linewidth=1.0, zorder=0)
    ax.set_xlabel("SHAP contribution (log-odds)", fontsize=10, color="#6c757d")
    ax.tick_params(axis="x", colors="#6c757d", labelsize=9)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#dee2e6")

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#e74c3c", label="Increases risk"),
              Patch(color="#27ae60", label="Decreases risk")]
    ax.legend(handles=legend, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    plt.tight_layout()
    return fig


def fraud_score_gauge(score: float) -> plt.Figure:
    """Minimal horizontal score bar."""
    fig, ax = plt.subplots(figsize=(5, 0.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color = "#e74c3c" if score >= 0.70 else ("#f39c12" if score >= 0.40 else "#27ae60")
    ax.barh([0], [1.0], color="#e9ecef", height=0.6)
    ax.barh([0], [score], color=color, height=0.6, alpha=0.85)

    ax.set_xlim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────

def shap_to_plain_english(shap_features: list[dict]) -> str:
    """Convert SHAP values to simple English summary."""
    if not shap_features:
        return "No significant feature contributions detected."

    risk_factors = [f for f in shap_features if f["shap_value"] > 0]
    safety_factors = [f for f in shap_features if f["shap_value"] < 0]

    parts = []

    if risk_factors:
        risk_text = ", ".join([f['feature'] for f in risk_factors[:3]])
        parts.append(f"**Risk signals:** {risk_text}")

    if safety_factors:
        safety_text = ", ".join([f['feature'] for f in safety_factors[:3]])
        parts.append(f"**Protective factors:** {safety_text}")

    return " | ".join(parts) if parts else "Balanced risk profile."


def agent_summary_plain_english(agent_output: dict) -> str:
    """Extract key points from agent output in plain English."""
    if not agent_output:
        return "No investigation results available."

    parts = []

    # Risk assessment
    risk = agent_output.get("risk_assessment", "unknown").upper()
    priority = agent_output.get("priority_score", 5)
    parts.append(f"**Assessment:** {risk} risk (Priority {priority}/10)")

    # Action
    action = agent_output.get("recommended_action", "unknown")
    action_map = {
        "auto_approve": "✓ Recommended to approve",
        "escalate": "⚠ Needs human review",
        "auto_decline": "✗ Recommended to decline"
    }
    parts.append(f"**Recommendation:** {action_map.get(action, action)}")

    # Reason codes if present
    codes = agent_output.get("adverse_action_reason_codes", [])
    if codes:
        code_text = ", ".join(codes[:3])
        parts.append(f"**Reason codes:** {code_text}")

    return " | ".join(parts)


def risk_badge(risk: str) -> str:
    cls = f"badge-{risk.lower()}"
    return f'<span class="badge {cls}">{risk.upper()}</span>'


def action_block(action: str) -> str:
    css = {"auto_approve": "approve", "escalate": "escalate", "auto_decline": "decline"}
    cls = f"action-{css.get(action, 'escalate')}"
    label = ACTION_LABELS.get(action, action)
    return f'<div class="action-block {cls}">{label}</div>'


def guardrail_html(check_name: str, result: dict) -> str:
    passed = result.get("passed", False)
    cls = "guardrail-pass" if passed else "guardrail-fail"
    status = "PASS" if passed else "FAIL"
    label_map = {
        "schema_validation": "Schema",
        "shap_consistency": "SHAP Grounding",
        "pii_leakage": "PII Check",
        "action_consistency": "Action Logic",
    }
    label = label_map.get(check_name, check_name)
    msg = result.get("message", "")[:80]
    return f"""
    <div class="guardrail-card {cls}">
        <div class="guardrail-label">{label}</div>
        <div class="guardrail-status">{status}</div>
        <div class="guardrail-message">{msg}</div>
    </div>"""


def detail_row(key: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        val_str = '<span style="color:#adb5bd">—</span>'
    elif isinstance(value, float):
        val_str = f"${value:,.2f}" if "Amt" in key else f"{value:.4f}"
    else:
        val_str = str(value)
    return f'<div class="detail-row"><span class="detail-key">{key}</span><span class="detail-value">{val_str}</span></div>'


def tooltip(label: str, explanation: str) -> str:
    """Create a label with hover tooltip."""
    return f'<span title="{explanation}" style="cursor:help; border-bottom:1px dotted #999;">{label}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size:18px; font-weight:700; color:#e8edf5; letter-spacing:-0.01em;">
            Fraud Investigation
        </div>
        <div style="font-size:11px; color:#7a95b8; margin-top:4px; font-weight:500; letter-spacing:0.05em; text-transform:uppercase;">
            Model Risk Management Demo
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:16px;">
        <div style="font-size:11px; font-weight:700; color:#7a95b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
            Pick a Transaction
        </div>
        <div style="font-size:12px; color:#b0bcc8; line-height:1.5;">
            Select from 30 real fraud investigation cases. Each has transaction details, model score, and AI investigation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    eval_cases = load_eval_cases()

    SCENARIO_ABBREV = {
        "clear_fraud": "FRAUD",
        "clear_legitimate": "LEGIT",
        "ambiguous_middle": "AMBIG",
        "adversarial_high_amount_clean": "ADV",
        "adversarial_high_score_fp": "ADV",
        "adversarial_low_score_fn": "ADV",
        "adversarial_nighttime_legit": "ADV",
        "adversarial_mid_score_fraud": "ADV",
    }

    if eval_cases:
        # Default: show simple dropdown
        options = []
        for c in eval_cases:
            abbrev = SCENARIO_ABBREV.get(c["scenario_type"], "CASE")
            score_str = f"{c['model_score']:.3f}"
            label = f"[{abbrev} {score_str}]  {c['eval_id'].upper()}"
            options.append(label)

        selected_idx = st.selectbox(
            "Case",
            range(len(options)),
            format_func=lambda i: options[i],
            label_visibility="collapsed",
            key="main_case_select",
        )
        selected_case = eval_cases[selected_idx]

        # Advanced filters (collapsed by default): search by ID + score range
        with st.expander("Advanced: Search & Filter", expanded=False):
            st.markdown('<div style="font-size:12px; color:#b0bcc8; margin-bottom:10px; line-height:1.5;">Narrow the 30 cases by typing part of a case ID, or drag the slider to a fraud-score band. A filtered dropdown appears below.</div>', unsafe_allow_html=True)

            st.markdown('<div style="font-size:11px; color:#7a95b8; margin:2px 0 2px;">Search by case ID (eval_001 – eval_030)</div>', unsafe_allow_html=True)
            search_query = st.text_input(
                "Search case ID", placeholder="e.g., eval_027",
                label_visibility="collapsed", key="case_search",
            )

            st.markdown('<div style="font-size:11px; color:#7a95b8; margin:10px 0 2px;">Fraud score range (0.00 = safe, 1.00 = certain fraud)</div>', unsafe_allow_html=True)
            min_score, max_score = st.slider(
                "Score range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                label_visibility="collapsed",
            )

            # Apply filters
            filtered_cases = [
                c for c in eval_cases
                if (not search_query or search_query.lower() in c["eval_id"].lower())
                and (min_score <= c["model_score"] <= max_score)
            ]

            if filtered_cases and len(filtered_cases) < len(eval_cases):
                st.info(f"Showing {len(filtered_cases)} of {len(eval_cases)} cases")
                filtered_options = []
                for c in filtered_cases:
                    abbrev = SCENARIO_ABBREV.get(c["scenario_type"], "CASE")
                    score_str = f"{c['model_score']:.3f}"
                    label = f"[{abbrev} {score_str}]  {c['eval_id'].upper()}"
                    filtered_options.append(label)
                filtered_idx = st.selectbox(
                    "Filtered cases",
                    range(len(filtered_options)),
                    format_func=lambda i: filtered_options[i],
                    label_visibility="collapsed",
                    key="filtered_case_select",
                )
                selected_case = filtered_cases[filtered_idx]
            elif not filtered_cases:
                st.warning("No cases match these filters.")
    else:
        st.warning("Eval set not found. Run: python scripts/generate_eval_set.py")
        st.stop()

    # Case metadata
    c = selected_case
    adv_tag = "  [Adversarial]" if c.get("adversarial") else ""
    st.markdown(f"""
    <div style="margin-top:16px; padding:14px; background:#1a2e4a; border-radius:8px; border:1px solid #2d4a6e;">
        <div style="font-size:11px; color:#7a95b8; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:10px;">Case Details</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:12px; color:#c9d6e8;">Transaction ID</span>
            <span style="font-size:12px; color:#e8edf5; font-weight:700;">{c['transaction_id']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:12px; color:#c9d6e8;">Model Score</span>
            <span style="font-size:12px; color:#e8edf5; font-weight:700;">{c['model_score']:.4f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:12px; color:#c9d6e8;">Actual Fraud</span>
            <span style="font-size:12px; color:{'#e74c3c' if c['actual_fraud'] else '#27ae60'}; font-weight:700;">{"Yes" if c['actual_fraud'] else "No"}</span>
        </div>
        <div style="margin-top:8px; font-size:11.5px; color:#8fa8c8; line-height:1.5;">{c['scenario_description'][:100]}</div>
        {"<div style='margin-top:6px; font-size:11px; color:#e67e22; font-weight:600;'>ADVERSARIAL CASE</div>" if c.get("adversarial") else ""}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:24px; border-top:1px solid #1e3050; padding-top:20px;"></div>', unsafe_allow_html=True)

    # Model stats from saved metrics
    champ_metrics_path = RESULTS_DIR / "champion_metrics.json"
    if champ_metrics_path.exists():
        champ = json.loads(champ_metrics_path.read_text())
        m = champ.get("val_metrics", {})
        st.markdown('<div style="font-size:11px; font-weight:700; color:#7a95b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">Champion Model</div>', unsafe_allow_html=True)
        for label, key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("FDR @ 3%", "fdr_at_3pct_review")]:
            val = m.get(key, 0)
            val_str = f"{val:.4f}" if key != "fdr_at_3pct_review" else f"{val:.2%}"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-size:12px; color:#7a95b8;">{label}</span>
                <span style="font-size:12px; color:#e8edf5; font-weight:700;">{val_str}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:24px; border-top:1px solid #1e3050; padding-top:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px; color:#4a6685; line-height:1.5;">XGBoost champion · SHAP explanations · SR 11-7 aligned · Claude Sonnet agent</div>', unsafe_allow_html=True)


# ── Load data for selected case ────────────────────────────────────────────

txn_id = selected_case["transaction_id"]

# Load cached agent result first — in demo mode it is the sole data source.
cached_result = load_cached_result(selected_case["eval_id"])

if DEMO_MODE:
    # Render entirely from the cached eval result (no model / parquets needed).
    txn = extract_tool_output(cached_result, "get_transaction_details")
    shap_data = extract_tool_output(cached_result, "get_shap_explanation")
    history = extract_tool_output(cached_result, "get_customer_history")
else:
    ctx = load_context()  # triggers model load (cached after first call)
    with st.spinner("Loading transaction data…"):
        txn = fetch_txn_details(txn_id)
        shap_data = fetch_shap(txn_id)
    card_id = txn.get("card1")
    history = fetch_history(int(card_id)) if card_id else {}


# ── Intro & Help ───────────────────────────────────────────────────────────

with st.expander("What is this tool?", expanded=False):
    st.markdown("""
    **Fraud Investigation System** — A decision-support tool that combines machine learning and AI reasoning to investigate suspicious transactions.

    **How it works:**
    1. You select a transaction from the dropdown (left sidebar)
    2. The system shows you:
       - **Left panel**: Transaction details, customer history, and a SHAP chart explaining which factors increase or decrease fraud risk
       - **Right panel**: An AI agent's investigation narrative, recommended action, and supporting reason codes
       - **Bottom**: Automated safety checks (guardrails) ensuring the agent output is valid and safe

    **Key concept — SHAP (SHapley Additive exPlanations):**
    The red and green bar chart shows which transaction features pushed the fraud score up (red) or down (green).
    Example: High transaction amount pushes fraud probability up; customer's clean history pushes it down.

    **Key concept — Priority Score (1–10):**
    How urgent this case is. 10 = highest priority (almost certainly fraud); 1 = lowest priority (almost certainly legitimate).

    **Key concept — Guardrails:**
    Four automated checks ensure the AI agent is not hallucinating, leaking personal information, or violating business rules.
    All must pass before the output is trusted.

    **This is a demo** using 30 real transactions from a fraud detection dataset. The agent uses Claude AI to synthesize evidence.
    """)

st.markdown(f"""
<div style="background:#f8f9fc; border-left:4px solid #2980b9; border-radius:0 8px 8px 0; padding:12px 18px; margin-bottom:20px;">
    <div style="font-size:13px; color:#2c3e50; line-height:1.6;">
        <strong>Using this tool:</strong> Select a transaction case above, then read left to right.
        Transaction details (left) provide context. SHAP chart shows evidence. Agent narrative (right) synthesizes the investigation.
        Guardrails (bottom) validate safety.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────

score = txn.get("fraud_score", selected_case["model_score"])
risk = "high" if score >= 0.70 else ("medium" if score >= 0.40 else "low")

col_h1, col_h2, col_h3, col_h4 = st.columns([3, 1, 1, 1])
with col_h1:
    st.markdown(f"""
    <div style="padding: 4px 0 16px 0;">
        <div style="font-size:22px; font-weight:700; color:#1a1a2e; letter-spacing:-0.02em;">
            Transaction {txn_id}
        </div>
        <div style="font-size:13px; color:#6c757d; margin-top:4px;">
            {selected_case['scenario_type'].replace('_', ' ').title()}
            {' — Adversarial Case' if selected_case.get('adversarial') else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.metric("Fraud Score", f"{score:.4f}")
    st.caption("Model confidence (0–1)")

with col_h3:
    st.metric("Risk Level", risk.upper())
    st.caption(f"{'High: likely fraud' if risk == 'high' else ('Medium: review needed' if risk == 'medium' else 'Low: likely legitimate')}")

with col_h4:
    st.metric("Actual Fraud", "Yes" if selected_case["actual_fraud"] else "No")
    st.caption("Ground truth for eval")

st.markdown("---")


# ── Main columns ───────────────────────────────────────────────────────────

col_left, col_right = st.columns([11, 13], gap="large")


# ════════════════════════════════ LEFT PANEL ════════════════════════════════

with col_left:

    # ── Score bar ──────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Fraud Probability Score</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; color:#666; margin-bottom:12px;">XGBoost model confidence: how likely this transaction is fraudulent (0–1 scale). Red = fraud risk, green = legitimate.</div>', unsafe_allow_html=True)
    gauge_fig = fraud_score_gauge(score)
    st.pyplot(gauge_fig, width='stretch')
    plt.close(gauge_fig)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f'<div style="font-size:11px; color:#6c757d; text-align:center;">0.00<br><span style="font-size:10px;">No risk</span></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div style="font-size:13px; font-weight:800; text-align:center; color:{RISK_COLORS[risk]};">{score:.4f}</div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div style="font-size:11px; color:#6c757d; text-align:center;">1.00<br><span style="font-size:10px;">Certain fraud</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Transaction details ────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Transaction Details</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; color:#666; margin-bottom:10px;">Key fields from the transaction. Amount, card type, timing, and location help the model assess risk.</div>', unsafe_allow_html=True)

    FIELD_LABELS = {
        "TransactionAmt": "Amount",
        "ProductCD": "Product Code",
        "card4": "Card Network",
        "card6": "Card Type",
        "P_emaildomain": "Purchaser Email Domain",
        "R_emaildomain": "Recipient Email Domain",
        "tx_hour": "Hour of Day",
        "tx_dayofweek": "Day of Week",
        "addr1": "Billing Address Code",
        "dist1": "Distance (billing-IP)",
        "M4": "Match Flag M4",
        "M5": "Match Flag M5",
        "M6": "Match Flag M6",
    }

    rows_html = ""
    for field, label in FIELD_LABELS.items():
        val = txn.get(field)
        if field == "TransactionAmt" and val is not None:
            display = f"${val:,.2f}"
        elif field == "tx_hour" and val is not None:
            h = int(val)
            m = int((val - h) * 60)
            display = f"{h:02d}:{m:02d}"
        elif val is None:
            display = None
        else:
            display = str(val)
        rows_html += detail_row(label, display)
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── What If Mode ───────────────────────────────────────────────────────
    if not DEMO_MODE:
      with st.expander("What If — Modify & Recalculate", expanded=False):
        st.markdown('<div style="font-size:12px; color:#555; margin-bottom:12px;">Adjust transaction details below to see how the fraud score would change.</div>', unsafe_allow_html=True)

        col_w1, col_w2 = st.columns(2)
        modifications = {}

        with col_w1:
            new_amount = st.number_input(
                "Transaction Amount ($)",
                value=float(txn.get("TransactionAmt", 100)),
                min_value=0.0,
                step=10.0,
                key="what_if_amount"
            )
            if new_amount != txn.get("TransactionAmt", 100):
                modifications["TransactionAmt"] = new_amount

            new_hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=int(txn.get("tx_hour", 12)),
                key="what_if_hour"
            )
            if new_hour != int(txn.get("tx_hour", 12)):
                modifications["tx_hour"] = float(new_hour)

        with col_w2:
            new_card_net = st.selectbox(
                "Card Network (card4)",
                ["visa", "mastercard", "amex", "discover"],
                index=["visa", "mastercard", "amex", "discover"].index(str(txn.get("card4", "visa")).lower()),
                key="what_if_card4"
            )
            if new_card_net != str(txn.get("card4", "")).lower():
                modifications["card4"] = new_card_net

            new_product = st.selectbox(
                "Product Code",
                ["W", "H", "C", "S", "R"],
                index=["W", "H", "C", "S", "R"].index(str(txn.get("ProductCD", "W"))),
                key="what_if_product"
            )
            if new_product != str(txn.get("ProductCD", "")):
                modifications["ProductCD"] = new_product

        # Estimate what if impact
        if modifications:
            new_score, impact_narrative = estimate_what_if_impact(txn, modifications)
            old_score = txn.get("fraud_score", 0.5)
            delta = new_score - old_score

            # Display comparison
            col_cmp1, col_cmp2, col_cmp3 = st.columns(3)

            with col_cmp1:
                st.metric("Original Score", f"{old_score:.4f}")

            with col_cmp2:
                st.metric("Est. New Score", f"{new_score:.4f}")

            with col_cmp3:
                delta_pct = (delta / old_score * 100) if old_score > 0 else 0
                st.metric("Est. Change", f"{delta:+.4f} ({delta_pct:+.1f}%)")

            # Impact explanation
            st.markdown(f'<div style="background:#f0f7ff; padding:10px 14px; border-radius:6px; border:1px solid #aed6f1; font-size:12px; color:#1a5276;"><strong>Impact analysis:</strong> {impact_narrative}</div>', unsafe_allow_html=True)

            # Risk assessment
            old_risk = "high" if old_score >= 0.70 else ("medium" if old_score >= 0.40 else "low")
            new_risk = "high" if new_score >= 0.70 else ("medium" if new_score >= 0.40 else "low")

            if old_risk != new_risk:
                st.markdown(f'<div style="background:#fff3cd; margin-top:8px; padding:10px 14px; border-radius:6px; border:1px solid #ffc107; font-size:12px; color:#856404;"><strong>Risk category change:</strong> {old_risk.upper()} → {new_risk.upper()}</div>', unsafe_allow_html=True)

            # Disclaimer
            st.markdown('<div style="margin-top:12px; font-size:11px; color:#999; border-top:1px solid #f0f0f0; padding-top:8px;"><em>Estimate based on feature direction change. Full accuracy requires re-running the complete feature engineering pipeline on raw transaction data.</em></div>', unsafe_allow_html=True)
        else:
            st.info("Adjust a field above to see how the score changes.")

    # ── Customer history ───────────────────────────────────────────────────
    if history and "n_transactions_in_window" in history:
        st.markdown('<div class="card"><div class="card-title">Card History (30-day window)</div>', unsafe_allow_html=True)
        n_txn = history.get("n_transactions_in_window", 0)
        if n_txn == 0:
            st.markdown('<div style="font-size:13px; color:#6c757d;">No transactions in the past 30 days.</div>', unsafe_allow_html=True)
        else:
            h_rows = ""
            h_rows += detail_row("Transactions", str(n_txn))
            h_rows += detail_row("Total Spend", f"${history.get('total_amount', 0):,.2f}")
            h_rows += detail_row("Mean Amount", f"${history.get('mean_amount', 0):,.2f}")
            h_rows += detail_row("Max Amount", f"${history.get('max_amount', 0):,.2f}")
            domains = history.get("unique_email_domains", [])
            h_rows += detail_row("Unique Email Domains", str(len(domains)))
            st.markdown(h_rows, unsafe_allow_html=True)
            if domains:
                st.markdown(f'<div style="margin-top:8px; font-size:12px; color:#6c757d;">{", ".join(str(d) for d in domains[:5])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP chart ─────────────────────────────────────────────────────────
    shap_features = shap_data.get("top_shap_features", []) if "error" not in shap_data else []
    if shap_features:
        st.markdown('<div class="card"><div class="card-title">SHAP Feature Contributions</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px; color:#666; margin-bottom:10px;">Why the model assigned this score: which transaction details pushed fraud risk up (red) or down (green).</div>', unsafe_allow_html=True)
        shap_fig = shap_bar_chart(shap_features)
        st.pyplot(shap_fig, width='stretch')
        plt.close(shap_fig)
        st.markdown('<div style="font-size:11px; color:#9ca3af; margin-top:6px;">Red bars increase fraud probability · Green bars decrease it · Units: log-odds</div>', unsafe_allow_html=True)

        # Feature glossary
        with st.expander("What do these features mean?", expanded=False):
            st.markdown("""
            **Feature groups:**
            - **V-features** (V1, V45, V258, etc.) — Vesta fraud detection signals. Proprietary indicators of fraud patterns.
            - **C-features** (C1, C13, C14, etc.) — Card and customer attributes. Card type, product category, email domain, network patterns.
            - **M-features** (M1–M9) — Match/identity verification flags. Address matches, email consistency, etc.
            - **D-features** — Other derived signals. Time, distance, and behavioral anomalies.

            **Key insight:** You don't need to know exactly what V258 means. What matters is:
            - **Red bar** = This factor made fraud risk go UP
            - **Green bar** = This factor made fraud risk go DOWN
            - The "In Plain English" section below translates the most important ones into business terms.
            """)

        st.markdown('</div>', unsafe_allow_html=True)

        # Plain English summary
        plain_summary = shap_to_plain_english(shap_features)
        st.markdown(f'<div class="card"><div class="card-title">In Plain English</div><div style="font-size:13.5px; color:#2c3e50; line-height:1.6;">{plain_summary}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════ RIGHT PANEL ════════════════════════════════

with col_right:

    # ── Agent investigation controls ────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Agent Investigation</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; color:#666; margin-bottom:10px;">Claude AI analyzes the transaction and model signal to make a recommendation. Grounded in facts, not hallucinations.</div>', unsafe_allow_html=True)

    agent_output = None
    guardrail_results = {}
    tool_log = []

    if cached_result and cached_result.get("output"):
        agent_output = cached_result["output"]
        guardrail_results = cached_result.get("guardrails", {})
        tool_log = cached_result.get("tool_call_log", [])
        st.markdown('<div style="display:inline-block; background:#eaf5fb; color:#1a5276; border:1px solid #aed6f1; border-radius:4px; padding:3px 10px; font-size:11px; font-weight:600;">Cached result</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:13px; color:#6c757d; padding:8px 0;">No cached investigation for this case.</div>', unsafe_allow_html=True)

    if DEMO_MODE:
        st.markdown('<div style="font-size:11px; color:#9ca3af; margin-top:6px;">Live re-runs are disabled in the hosted demo. Results shown are pre-computed by the offline eval suite (<code>evals/run_evals.py</code>). Clone the repo and add an API key to run live.</div>', unsafe_allow_html=True)
        run_live = False
    else:
        from dotenv import load_dotenv
        load_dotenv()
        has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

        run_live = st.button(
            "Run Live Investigation" if not agent_output else "Re-run Live Investigation",
            disabled=not has_key,
            help="Requires ANTHROPIC_API_KEY in .env" if not has_key else None,
        )

    if run_live:
        from src.agent.investigator import investigate
        with st.spinner("Agent investigating — calling tools and synthesizing…"):
            result = investigate(transaction_id=txn_id)
        if result.error:
            st.error(f"Agent error: {result.error}")
        else:
            agent_output = result.raw_output
            guardrail_results = result.guardrail_results
            tool_log = result.tool_call_log
            # Save to cache
            cache_data = {
                "eval_id": selected_case["eval_id"],
                "transaction_id": txn_id,
                "output": agent_output,
                "guardrails": guardrail_results,
                "tool_call_log": tool_log,
                "iterations_used": result.iterations_used,
                "error": result.error,
            }
            (EVAL_CACHE_DIR / f"{selected_case['eval_id']}.json").write_text(
                json.dumps(cache_data, indent=2, default=str)
            )
            st.success(f"Investigation complete — {result.iterations_used} tool-call rounds")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Agent output ────────────────────────────────────────────────────────
    if agent_output:
        risk_out = agent_output.get("risk_assessment", "medium")
        action_out = agent_output.get("recommended_action", "escalate")
        priority = agent_output.get("priority_score", 5)
        narrative = agent_output.get("sar_narrative", "")
        codes = agent_output.get("adverse_action_reason_codes", [])
        evidence = agent_output.get("supporting_evidence", [])

        # ── Decision summary ────────────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">Decision Summary</div>', unsafe_allow_html=True)
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Risk Assessment", risk_out.upper())
        with dcol2:
            st.metric("Priority Score", f"{priority} / 10")
        with dcol3:
            st.metric("Tool Rounds", str(len(set(l["tool"] for l in tool_log))))
        st.markdown(f'<div style="margin-top:12px;">{action_block(action_out)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Plain English summary ───────────────────────────────────────────
        plain_summary = agent_summary_plain_english(agent_output)
        st.markdown(f'<div class="card"><div class="card-title">Quick Summary</div><div style="font-size:13px; color:#2c3e50; line-height:1.6;">{plain_summary}</div></div>', unsafe_allow_html=True)

        # ── SAR narrative ────────────────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">SAR Narrative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sar-box">{narrative}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Reason codes ─────────────────────────────────────────────────────
        if codes:
            st.markdown('<div class="card"><div class="card-title">Adverse Action Reason Codes</div>', unsafe_allow_html=True)
            # Prefer descriptions embedded in the cached reason_codes tool output
            # (keeps the hosted demo free of heavy imports); fall back to the source map.
            REASON_CODES = {
                c.get("code"): c.get("description", "")
                for c in extract_tool_output(cached_result, "get_reason_codes").get("adverse_action_codes", [])
            }
            if not REASON_CODES:
                try:
                    from src.explain.shap_utils import REASON_CODES
                except Exception:
                    REASON_CODES = {}
            pills = ""
            for code in codes:
                desc = REASON_CODES.get(code, "")
                pills += f'<span class="code-pill" title="{desc}">{code}</span>'
            st.markdown(pills, unsafe_allow_html=True)
            if codes:
                st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)
                for code in codes:
                    desc = REASON_CODES.get(code, "")
                    if desc:
                        st.markdown(f'<div style="font-size:12px; color:#555; margin-bottom:3px;"><strong style="color:#1a5276">{code}</strong> — {desc}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Supporting evidence ───────────────────────────────────────────────
        if evidence:
            with st.expander(f"Supporting Evidence  ({len(evidence)} items)", expanded=False):
                for i, item in enumerate(evidence, 1):
                    st.markdown(f"""
                    <div style="margin-bottom:12px; padding:10px 14px; background:#f8f9fc; border-radius:6px; border-left:2px solid #aed6f1;">
                        <div style="font-size:13px; color:#1a1a2e; font-weight:500; margin-bottom:4px;">{i}. {item.get('claim', '')}</div>
                        <div style="font-size:11.5px; color:#6c757d;"><strong>Source:</strong> {item.get('source', '')} &nbsp;|&nbsp; <strong>Value:</strong> {item.get('value', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tool call log ─────────────────────────────────────────────────────
        if tool_log:
            with st.expander("Tool Call Log", expanded=False):
                for entry in tool_log:
                    st.markdown(f"""
                    <div style="font-size:12px; padding:6px 0; border-bottom:1px solid #f0f2f5; color:#374151;">
                        <strong style="color:#1a5276;">iter {entry['iteration']}</strong> &nbsp;
                        <code style="background:#f0f4ff; padding:2px 6px; border-radius:3px; font-size:11px;">{entry['tool']}</code> &nbsp;
                        <span style="color:#6c757d;">{json.dumps(entry['input'])}</span>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:40px 0; text-align:center; color:#9ca3af;">
            <div style="font-size:14px; font-weight:600; margin-bottom:8px;">No investigation results</div>
            <div style="font-size:13px;">Run the eval suite or click "Run Live Investigation" above.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Guardrail results ──────────────────────────────────────────────────────

if guardrail_results:
    st.markdown("---")
    st.markdown('<div style="font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#6c757d; margin-bottom:8px;">Guardrail Validation</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; color:#666; margin-bottom:12px;">Safety checks ensure the AI output is valid, consistent, and doesn\'t leak personal information. All must pass for analyst review.</div>', unsafe_allow_html=True)

    CHECK_ORDER = ["schema_validation", "shap_consistency", "pii_leakage", "action_consistency"]
    gcols = st.columns(4)
    for col, check in zip(gcols, CHECK_ORDER):
        result = guardrail_results.get(check, {"passed": False, "message": "Not run"})
        with col:
            st.markdown(guardrail_html(check, result), unsafe_allow_html=True)

    all_pass = all(guardrail_results.get(c, {}).get("passed", False) for c in CHECK_ORDER)
    st.markdown(f"""
    <div style="margin-top:12px; text-align:right; font-size:12px; color:{'#196f3d' if all_pass else '#c0392b'}; font-weight:700;">
        {'All guardrails passed' if all_pass else 'One or more guardrails failed — analyst review required'}
    </div>
    """, unsafe_allow_html=True)
