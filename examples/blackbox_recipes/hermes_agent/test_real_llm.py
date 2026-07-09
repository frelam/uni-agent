"""Real LLM smoke test for the Hermes agent entrypoint.

Points the entrypoint at a real OpenAI-compatible API and runs a simple
tool-use task. This validates the full loop: model → Hermes tool calls →
parsing → tool execution → observation → model continuation.

Usage:
    # Uses the hermes CLI's configured model by default
    python examples/blackbox_recipes/hermes_agent/test_real_llm.py

    # Or specify explicitly:
    HERMES_BASE_URL=https://api.deepseek.com/v1 \
    HERMES_MODEL=deepseek-v4-flash \
    python examples/blackbox_recipes/hermes_agent/test_real_llm.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Read config from the same source as hermes CLI
def _read_hermes_config():
    """Read ~/.hermes/config.yaml for model/base_url configuration."""
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = _read_hermes_config()
    model_cfg = config.get("model", {})

    base_url = os.environ.get("HERMES_BASE_URL") or model_cfg.get("base_url", "")
    model = os.environ.get("HERMES_MODEL") or model_cfg.get("default", "")
    # DeepSeek key: check env vars that hermes or the user might set
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("HERMES_API_KEY") or model_cfg.get("api_key", "") or ""

    if not base_url:
        print("ERROR: No base_url configured. Set HERMES_BASE_URL or configure ~/.hermes/config.yaml")
        return 1

    print(f"=== Hermes Agent Real LLM Test ===")
    print(f"Base URL:  {base_url}")
    print(f"Model:     {model}")
    print()

    # Add project root to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from examples.blackbox_recipes.hermes_agent.run_hermes_in_sandbox import (
        run_agent,
        build_system_prompt,
        chat_completions,
    )

    # Simple tools for the test
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a bash command and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit the final answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Summary of what was done"}
                    },
                    "required": ["message"],
                },
            },
        },
    ]

    task = (
        "Please do the following:\n"
        "1. List files in /tmp\n"
        "2. Run 'python --version' to check the Python version\n"
        "3. Report both outputs, then submit"
    )

    print(f"Task: {task}")
    print(f"{'─' * 60}")

    started = time.time()
    try:
        run_agent(base_url, task, tools, max_turns=5, api_key=api_key)
    except Exception as e:
        print(f"\nERROR: Agent loop failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - started
    print(f"{'─' * 60}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
