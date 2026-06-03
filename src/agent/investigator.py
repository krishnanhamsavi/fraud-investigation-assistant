"""Agentic investigation loop — Anthropic tool-use, no framework dependencies.

Architecture:
  1. Build initial user message with the transaction ID to investigate.
  2. Loop: call Claude → execute any tool_use blocks → feed results back.
  3. When Claude returns stop_reason == "end_turn", extract the JSON report.
  4. Run guardrails on the report.
  5. Return structured InvestigationResult.

No LangChain, LangGraph, or CrewAI — raw Anthropic SDK messages API only.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from src.agent.tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_ITERATIONS = 6          # hard cap on tool-call rounds before forcing output
PROMPT_PATH = Path(__file__).parent / "prompts" / "v1.txt"

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InvestigationResult:
    transaction_id: int
    raw_output: dict                              # parsed agent JSON
    guardrail_results: dict = field(default_factory=dict)
    tool_call_log: list[dict] = field(default_factory=list)
    iterations_used: int = 0
    error: str | None = None

    @property
    def passed_all_guardrails(self) -> bool:
        if not self.guardrail_results:
            return False
        return all(v.get("passed", False) for v in self.guardrail_results.values())


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first well-formed JSON object from the agent's text response.

    Handles bare JSON, markdown code fences, and leading/trailing prose by
    finding the first '{' then walking forward with a depth counter — correctly
    handles nested objects and ignores braces inside string literals.
    """
    # Try stripping markdown fences first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Find the FIRST '{' then walk forward to its matching '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in agent response")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])

    raise ValueError("Unmatched braces — could not extract JSON from agent response")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def investigate(
    transaction_id: int,
    api_key: str | None = None,
) -> InvestigationResult:
    """Run the full investigation loop for a single flagged transaction.

    Parameters
    ----------
    transaction_id : the TransactionID to investigate (must exist in val split)
    api_key        : Anthropic API key; falls back to ANTHROPIC_API_KEY env var

    Returns
    -------
    InvestigationResult with parsed output, guardrail results, and tool log
    """
    from src.agent.guardrails import run_all_guardrails

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Create a .env file with your key "
            "or set the environment variable before running."
        )

    client = anthropic.Anthropic(api_key=key)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Please investigate transaction ID {transaction_id}. "
                f"Follow the workflow in your instructions: call each tool in order, "
                f"then produce your final JSON report."
            ),
        }
    ]

    tool_call_log: list[dict] = []
    iterations = 0
    raw_output: dict | None = None
    error: str | None = None

    for iteration in range(MAX_ITERATIONS + 1):
        iterations = iteration + 1

        # Force final answer on last iteration
        tool_choice: dict | str
        if iteration >= MAX_ITERATIONS:
            logger.warning(
                "Reached max iterations (%d) — forcing final answer.", MAX_ITERATIONS
            )
            messages.append({
                "role": "user",
                "content": (
                    "You have used the maximum number of tool calls. "
                    "Based on the evidence collected so far, produce your final "
                    "JSON report now. No further tool calls."
                ),
            })
            tool_choice = {"type": "none"}
        else:
            tool_choice = {"type": "auto"}

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice=tool_choice,
            messages=messages,
        )

        logger.debug(
            "Iteration %d — stop_reason: %s, blocks: %d",
            iterations, response.stop_reason,
            len(response.content),
        )

        # Add assistant turn to messages
        messages.append({"role": "assistant", "content": response.content})

        # ---- Tool use ----
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input))
                result = execute_tool(block.name, block.input)
                tool_call_log.append({
                    "iteration": iterations,
                    "tool": block.name,
                    "input": block.input,
                    "output": result,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_results})
            continue  # next iteration

        # ---- End turn — extract JSON ----
        if response.stop_reason == "end_turn":
            text_blocks = [b for b in response.content if b.type == "text"]
            if not text_blocks:
                error = "Agent returned end_turn with no text blocks"
                break

            full_text = "\n".join(b.text for b in text_blocks)
            try:
                raw_output = _extract_json(full_text)
                logger.info(
                    "Investigation complete — risk: %s, action: %s",
                    raw_output.get("risk_assessment"),
                    raw_output.get("recommended_action"),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"Failed to parse agent JSON: {exc}\n\nRaw text:\n{full_text}"
                logger.error(error)
            break

        # Unexpected stop reason
        error = f"Unexpected stop_reason: {response.stop_reason!r}"
        break

    if raw_output is None and error is None:
        error = "Agent loop exited without producing output"

    # ---- Guardrails ----
    guardrail_results: dict = {}
    if raw_output is not None:
        shap_data = next(
            (log["output"] for log in tool_call_log if log["tool"] == "get_shap_explanation"),
            None,
        )
        guardrail_results = run_all_guardrails(raw_output, shap_data)

    return InvestigationResult(
        transaction_id=transaction_id,
        raw_output=raw_output or {},
        guardrail_results=guardrail_results,
        tool_call_log=tool_call_log,
        iterations_used=iterations,
        error=error,
    )
