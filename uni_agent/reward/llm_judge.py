"""LLM Judge reward plugin for blackbox agent training.

Provides a pluggable ``compute_score`` for verl's ``custom_reward_function``
pattern, plus inline LLM judge utilities.

Architecture::

    Dataset (per-sample scoring config):
        extra_info.tools_kwargs.scoring = {
            "sandbox_eval": True,   # run tests in sandbox?
            "llm_judge": True,      # call LLM judge?
        }

    Sandbox runner (data collection only):
        → runs agent in sandbox
        → runs sandbox tests (if ``sandbox_eval``)
        → posts raw data to Gateway (task, agent_output, sandbox results)

    ``compute_score`` (this module, called by RewardLoopWorker):
        → reads sandbox results from extra_info
        → calls LLM judge (if ``llm_judge``)
        → aggregates into final reward_score

Usage in training config::

    reward:
      custom_reward_function:
        path: uni_agent.reward.llm_judge
        name: compute_score

Configuration (via environment variables):
    JUDGE_MODEL       — model name (default: deepseek-chat)
    JUDGE_BASE_URL    — API base URL (default: https://api.deepseek.com)
    JUDGE_API_KEY     — API key
    JUDGE_PROMPTS_DIR — directory for dataset-specific judge prompts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Prompt loading ───────────────────────────────────────────────────────────

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

JUDGE_PROMPT_MAP: dict[str, str | None] = {
    "math": "math_judge.txt",
    "coding": "coding_judge.txt",
    "terminal": "terminal_judge.txt",
    "terminaltraj": "terminal_judge.txt",
    "swe_zero": "coding_judge.txt",
    "open_swe_traces": "coding_judge.txt",
    "toolmind": None,
}


def load_judge_prompt(data_source: str, prompts_dir: str | None = None) -> str | None:
    """Load a dataset-specific judge prompt.

    Looks for ``<prompts_dir>/<mapped_name>.txt`` based on ``data_source``.
    Falls back to ``JUDGE_PROMPTS_DIR`` env var, then ``uni_agent/reward/prompts/``.

    Returns ``None`` if no prompt file is found (caller should use default).
    """
    data_source = (data_source or "").strip().lower()
    if not data_source:
        return None

    filename = JUDGE_PROMPT_MAP.get(data_source)
    if filename is None:
        for key, fname in JUDGE_PROMPT_MAP.items():
            if fname and key in data_source:
                filename = fname
                break

    if filename is None:
        logger.info("No judge prompt configured for data_source=%r; using default rubric.", data_source)
        return None

    # Resolve prompts directory
    search_dirs: list[Path] = []
    if prompts_dir:
        search_dirs.append(Path(prompts_dir))
    env_dir = os.environ.get("JUDGE_PROMPTS_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    search_dirs.append(_DEFAULT_PROMPTS_DIR)

    for d in search_dirs:
        prompt_path = d / filename
        if prompt_path.is_file():
            logger.info("Loaded judge prompt for data_source=%r from %s", data_source, prompt_path)
            return prompt_path.read_text(encoding="utf-8")

    logger.warning("Judge prompt file %s not found in %s; using default rubric.", filename, search_dirs)
    return None


# ── Judge config ─────────────────────────────────────────────────────────────


def _judge_config() -> tuple[str, str, str]:
    """Return (model, base_url, api_key) from environment."""
    model = os.environ.get("JUDGE_MODEL", "deepseek-chat")
    base_url = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("JUDGE_API_KEY or DEEPSEEK_API_KEY environment variable is required for LLM judge")
    return model, base_url, api_key


# ── LLM Judge API ────────────────────────────────────────────────────────────


async def judge_single(
    task: str,
    agent_output: str,
    *,
    rubric: str | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Score a single agent trajectory with an LLM judge.

    Args:
        task: The original task / prompt.
        agent_output: The agent's stdout (final answer + tool logs).
        rubric: Custom scoring rubric.  If None, uses a default.
        model: Judge model name (overrides JUDGE_MODEL env).
        base_url: Judge API base URL (overrides JUDGE_BASE_URL env).
        api_key: Judge API key (overrides JUDGE_API_KEY env).

    Returns:
        dict with ``reward_score`` (float 0-1) and ``judge_reason`` (str).
    """
    _model, _base_url, _api_key = _judge_config()
    model = model or _model
    base_url = base_url or _base_url
    api_key = api_key or _api_key

    if rubric is None:
        rubric = (
            "Evaluate whether the agent successfully completed the task.\n"
            "Consider:\n"
            "1. Did the agent produce the correct output?\n"
            "2. Was the approach reasonable and efficient?\n"
            "3. Did the agent use tools appropriately?\n\n"
            "Score: 1.0 = fully correct, 0.5 = partially correct, 0.0 = incorrect."
        )

    judge_prompt = (
        f"## Task\n{task[:2000]}\n\n"
        f"## Agent Output\n{agent_output[:3000]}\n\n"
        f"## Scoring Rubric\n{rubric}\n\n"
        "Respond with a JSON object:\n"
        '{"score": <0.0-1.0 float>, "reason": "<brief explanation>"}'
    )

    system_content = system_prompt or "You are an expert evaluator. Always respond with valid JSON."

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": judge_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    judge_text = data["choices"][0]["message"]["content"]
    try:
        judge_result = json.loads(judge_text)
        score = float(judge_result.get("score", 0.5))
        reason = judge_result.get("reason", judge_text[:200])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.5
        reason = judge_text[:200]

    return {
        "reward_score": min(max(score, 0.0), 1.0),
        "judge_reason": reason,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate_scores(
    sandbox_score: float,
    llm_score: float | None,
    scoring: dict,
) -> float:
    """Combine sandbox and LLM judge scores into a final reward."""
    has_sandbox = scoring.get("sandbox_eval", True)
    has_llm = scoring.get("llm_judge", False)

    if has_sandbox and has_llm:
        weight_accuracy = scoring.get("sandbox_weight", 0.5)
        weight_process = 1.0 - weight_accuracy
        return weight_accuracy * sandbox_score + weight_process * (llm_score or 0.0)
    elif has_sandbox:
        return sandbox_score
    elif has_llm:
        return llm_score or 0.0
    return 0.0


# ── Verl plugin (custom_reward_function) ────────────────────────────────────


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Verl-compatible reward function for blackbox agent DPO.

    This is the pluggable ``custom_reward_function`` wired in the training
    config.  The sandbox runner posts raw data (task, agent_output, optional
    sandbox test results) to the Gateway via ``reward_info_url``.  The
    framework merges that into ``extra_info``, so this function reads the
    runner-produced data and applies the configured scoring pipeline.

    Scoring pipeline (controlled by ``extra_info.scoring``):

    ======================  ===========================================
    Config                   Behavior
    ======================  ===========================================
    ``sandbox_eval: True``   Reads ``accuracy_reward`` from extra_info
                             (pre-computed by runner in the sandbox)
    ``llm_judge: True``      Calls ``judge_single`` against the LLM
                             judge API (requires JUDGE_API_KEY)
    both                     Weighted sum (``sandbox_weight``, default 0.5)
    ======================  ===========================================

    Returns:
        dict with ``score`` (float) and per-dimension metrics.
    """
    extra_info = extra_info or {}
    scoring = extra_info.get("scoring", {})
    if not scoring:
        scoring = {"sandbox_eval": True, "llm_judge": False}

    sandbox_score: float = 0.0
    llm_score: float | None = None
    judge_reason: str = ""

    # ── Sandbox evaluation result (pre-computed by runner) ──────────
    if scoring.get("sandbox_eval", True):
        sandbox_score = float(extra_info.get("accuracy_reward", 0.0))
        logger.info(
            "compute_score: sandbox accuracy=%.2f (data_source=%s)",
            sandbox_score,
            data_source,
        )

    # ── LLM judge ───────────────────────────────────────────────────
    if scoring.get("llm_judge", False):
        task = extra_info.get("task", "")
        agent_output = extra_info.get("agent_output", "")
        prompts_dir = extra_info.get("judge_prompts_dir", None)

        try:
            rubric = load_judge_prompt(str(data_source), prompts_dir=prompts_dir)
        except Exception:
            rubric = None

        try:
            result = asyncio.run(
                judge_single(task=task, agent_output=agent_output, rubric=rubric)
            )
            llm_score = float(result.get("reward_score", 0.0))
            judge_reason = str(result.get("judge_reason", ""))
        except Exception:
            logger.warning("LLM judge failed for data_source=%s; defaulting to 0.5", data_source)
            llm_score = 0.5
            judge_reason = "LLM judge error"

        logger.info(
            "compute_score: LLM judge score=%.2f (data_source=%s)",
            llm_score,
            data_source,
        )

    # ── Aggregate ───────────────────────────────────────────────────
    final_score = _aggregate_scores(sandbox_score, llm_score, scoring)

    logger.info(
        "compute_score: final=%.2f (sandbox=%.2f, llm=%s, data_source=%s)",
        final_score,
        sandbox_score,
        str(llm_score),
        data_source,
    )

    return {
        "score": final_score,
        "accuracy_reward": sandbox_score,
        "process_reward": llm_score,
        "judge_reason": judge_reason,
    }
