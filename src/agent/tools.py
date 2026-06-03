"""Tool definitions and implementations for the fraud investigation agent.

Each tool has two representations:
  1. A Python function that actually executes (returns a dict).
  2. An entry in TOOLS (the Anthropic tool-use JSON schema format).

Data is loaded lazily on first call and cached for the session.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.explain.shap_utils import (
    REASON_CODES,
    feature_to_reason_code,
    get_top_reasons,
)
from src.models.champion import load_champion
from src.models.train import FEATURE_EXCLUDE

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# Human-readable fields returned to the agent (avoids overwhelming context)
DETAIL_FIELDS = [
    "TransactionID", "TransactionAmt", "TransactionDT",
    "ProductCD", "card1", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "addr1", "addr2", "dist1",
    "tx_hour", "tx_dayofweek",
    "M1", "M4", "M5", "M6",
]


# ---------------------------------------------------------------------------
# Agent context (lazy singleton)
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """All data and model objects needed by the tools."""
    val_encoded: pd.DataFrame          # processed val parquet (model input)
    val_raw: pd.DataFrame              # raw joined val data (human-readable)
    train_raw: pd.DataFrame            # raw joined train data (for history lookup)
    feature_cols: list[str]
    model: Any
    explainer: shap.TreeExplainer
    scores: np.ndarray                 # pre-scored val set


_ctx: AgentContext | None = None


def _load_context() -> AgentContext:
    logger.info("Loading agent context (first call — subsequent calls use cache) …")

    val_enc = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    train_enc = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    feature_cols = [c for c in val_enc.columns if c not in FEATURE_EXCLUDE]
    model = load_champion()
    explainer = shap.TreeExplainer(model)

    # Raw joined data — needed for human-readable field values
    joined = pd.read_parquet(PROCESSED_DIR / "joined.parquet")

    # Apply engineered features (same as preprocess pipeline)
    from src.data.preprocess import build_features
    joined = build_features(joined)

    boundary_dt = train_enc["TransactionDT"].max()
    train_raw = joined[joined["TransactionDT"] <= boundary_dt]
    val_raw = joined[joined["TransactionDT"] > boundary_dt]

    logger.info("Scoring val set for context …")
    scores = model.predict_proba(val_enc[feature_cols])[:, 1]

    return AgentContext(
        val_encoded=val_enc,
        val_raw=val_raw,
        train_raw=train_raw,
        feature_cols=feature_cols,
        model=model,
        explainer=explainer,
        scores=scores,
    )


def get_context() -> AgentContext:
    global _ctx
    if _ctx is None:
        _ctx = _load_context()
    return _ctx


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_transaction_details(transaction_id: int) -> dict:
    """Return selected fields for a single transaction (human-readable values)."""
    ctx = get_context()
    mask = ctx.val_raw["TransactionID"] == transaction_id
    if not mask.any():
        return {"error": f"TransactionID {transaction_id} not found in validation set"}

    row = ctx.val_raw[mask].iloc[0]
    result = {}
    for f in DETAIL_FIELDS:
        if f in row.index:
            val = row[f]
            result[f] = None if pd.isna(val) else (
                round(float(val), 4) if isinstance(val, float) else val
            )

    # Attach the model's fraud probability score
    enc_mask = ctx.val_encoded["TransactionID"] == transaction_id
    if enc_mask.any():
        idx = ctx.val_encoded[enc_mask].index[0]
        result["fraud_score"] = round(float(ctx.scores[
            ctx.val_encoded.index.get_loc(idx)
        ]), 4)

    return result


def get_shap_explanation(transaction_id: int, top_n: int = 6) -> dict:
    """Return top SHAP feature contributions for a single transaction."""
    ctx = get_context()
    mask = ctx.val_encoded["TransactionID"] == transaction_id
    if not mask.any():
        return {"error": f"TransactionID {transaction_id} not found"}

    try:
        reasons = get_top_reasons(
            transaction_id=transaction_id,
            explainer=ctx.explainer,
            df=ctx.val_encoded,
            feature_cols=ctx.feature_cols,
            n=top_n,
        )
        return {
            "transaction_id": transaction_id,
            "top_shap_features": reasons,
            "note": (
                "shap_value > 0 increases fraud probability; "
                "shap_value < 0 decreases it. "
                "Values are in log-odds space."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_customer_history(card_id: int | str, lookback_days: int = 30) -> dict:
    """Return aggregate transaction statistics for a card over a lookback window.

    Uses the training + validation data as a proxy for transaction history.
    card_id corresponds to the card1 field.
    lookback_days is relative to the LATEST transaction in the dataset.
    """
    ctx = get_context()
    all_txns = pd.concat([ctx.train_raw, ctx.val_raw], ignore_index=True)

    try:
        card_id_int = int(card_id)
    except (ValueError, TypeError):
        return {"error": f"Invalid card_id: {card_id!r}"}

    card_mask = all_txns["card1"] == card_id_int
    if not card_mask.any():
        return {
            "card_id": card_id_int,
            "n_transactions_in_window": 0,
            "note": "No transaction history found for this card.",
        }

    max_dt = all_txns["TransactionDT"].max()
    lookback_dt = max_dt - lookback_days * 86400

    history = all_txns[card_mask & (all_txns["TransactionDT"] >= lookback_dt)]

    if history.empty:
        return {
            "card_id": card_id_int,
            "n_transactions_in_window": 0,
            "note": f"No transactions in the last {lookback_days} days.",
        }

    return {
        "card_id": card_id_int,
        "lookback_days": lookback_days,
        "n_transactions_in_window": int(len(history)),
        "total_amount": round(float(history["TransactionAmt"].sum()), 2),
        "mean_amount": round(float(history["TransactionAmt"].mean()), 2),
        "max_amount": round(float(history["TransactionAmt"].max()), 2),
        "min_amount": round(float(history["TransactionAmt"].min()), 2),
        "unique_products": history["ProductCD"].dropna().unique().tolist(),
        "unique_email_domains": history["P_emaildomain"].dropna().unique().tolist()[:5],
        "hours_active": sorted(history["tx_hour"].dropna().round(0).astype(int).unique().tolist()),
    }


def get_reason_codes(transaction_id: int) -> dict:
    """Return ECOA adverse action reason codes for a transaction."""
    ctx = get_context()
    mask = ctx.val_encoded["TransactionID"] == transaction_id
    if not mask.any():
        return {"error": f"TransactionID {transaction_id} not found"}

    try:
        reasons = get_top_reasons(
            transaction_id=transaction_id,
            explainer=ctx.explainer,
            df=ctx.val_encoded,
            feature_cols=ctx.feature_cols,
            n=4,
        )
        # Deduplicate codes while preserving rank order
        seen: set[str] = set()
        codes = []
        for r in reasons:
            if r["reason_code"] not in seen and r["shap_value"] > 0:
                seen.add(r["reason_code"])
                codes.append({
                    "code": r["reason_code"],
                    "description": REASON_CODES[r["reason_code"]],
                    "driven_by_feature": r["feature"],
                })
        return {
            "transaction_id": transaction_id,
            "adverse_action_codes": codes,
            "note": "Codes reflect features that increase fraud probability (positive SHAP only).",
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Anthropic tool schema definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_transaction_details",
        "description": (
            "Retrieve selected fields for a flagged transaction, including amount, "
            "product type, card type, email domains, address, time-of-day, and the "
            "model's fraud probability score. Use this as the first step in every "
            "investigation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "integer",
                    "description": "The TransactionID to look up.",
                }
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "get_shap_explanation",
        "description": (
            "Return the top SHAP feature contributions that drove the fraud score for "
            "a specific transaction. Each entry shows the feature name, its value, "
            "its SHAP contribution (positive = increases fraud probability), and the "
            "direction. Use this to understand WHY the model flagged the transaction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "integer",
                    "description": "The TransactionID to explain.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top features to return (default 6, max 10).",
                    "default": 6,
                },
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "get_customer_history",
        "description": (
            "Return aggregate transaction statistics for a card (identified by card1 "
            "value) over a recent lookback window. Use this to assess whether the "
            "flagged transaction fits the card's established behavioral pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "integer",
                    "description": "The card1 value identifying the card.",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back (default 30).",
                    "default": 30,
                },
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "get_reason_codes",
        "description": (
            "Return ECOA-compliant adverse action reason codes for a transaction, "
            "derived from SHAP feature contributions. Use these codes in the "
            "adverse_action_reason_codes field of your final report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "integer",
                    "description": "The TransactionID to retrieve reason codes for.",
                }
            },
            "required": ["transaction_id"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_FN_MAP = {
    "get_transaction_details": get_transaction_details,
    "get_shap_explanation": get_shap_explanation,
    "get_customer_history": get_customer_history,
    "get_reason_codes": get_reason_codes,
}


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call by name and return the result dict."""
    fn = _TOOL_FN_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name!r}"}
    try:
        return fn(**args)
    except Exception as exc:
        logger.exception("Tool %r raised an exception", name)
        return {"error": f"Tool execution failed: {exc}"}
