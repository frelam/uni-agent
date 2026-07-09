"""Hermes-format agent entrypoint — runs inside the SWE-bench sandbox.

Drives a tool-use conversation loop against the Uni-Agent Gateway
(OpenAI-compatible ``/v1/chat/completions``), parsing Hermes-format tool
calls (``<tool_call>{"name": ..., "arguments": ...}</tool_call>``) and
executing them via subprocess inside the sandbox.

No dependencies beyond Python stdlib — runs with the sandbox's own Python.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hermes] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("hermes_entrypoint")

# ── Hermes tool-call format ─────────────────────────────────────────────────

_HERMES_START = "<tool_call>"
_HERMES_END = "</tool_call>"
_HERMES_PATTERN = re.compile(
    rf"{re.escape(_HERMES_START)}\s*(.*?)\s*{re.escape(_HERMES_END)}",
    re.DOTALL,
)

FINISH_TOOLS = frozenset({"finish", "submit", "stop", "exit"})


def parse_hermes_tool_calls(text: str, known_tools: list[dict]) -> list[dict]:
    """Extract Hermes-format tool calls from model output.

    Returns list of OpenAI-style tool_calls:
      [{"id": "call_N", "type": "function",
        "function": {"name": <str>, "arguments": <str>}}, ...]
    """
    tool_name_set = {t["function"]["name"] for t in known_tools}
    tool_calls = []
    for idx, match in enumerate(_HERMES_PATTERN.finditer(text)):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Hermes tool call JSON: %.200s", raw)
            continue
        name = parsed.get("name", "")
        if name not in tool_name_set:
            logger.warning("Unknown tool '%s', skipping. Known: %s", name, sorted(tool_name_set))
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    # Return text before first tool call as content
    content = text
    if tool_calls:
        first_start = text.find(_HERMES_START)
        if first_start >= 0:
            content = text[:first_start].strip()
    return content, tool_calls


# ── Tool execution ──────────────────────────────────────────────────────────


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool in the sandbox and return the observation string."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        args = {}

    if name in FINISH_TOOLS:
        return json.dumps({"status": "finished", "message": args.get("message", "")})

    if name == "execute_bash":
        command = args.get("command", "")
        if not command:
            return "Error: no command provided"
        return _run_bash(command)

    if name == "str_replace_editor":
        return _run_str_replace_editor(args)

    logger.warning("Unknown tool: %s", name)
    return f"Error: unknown tool '{name}'"


def _resolve_cwd() -> str:
    """Return the working directory for tool execution.

    Uses ``/testbed`` when available (SWE-bench sandbox), otherwise ``/tmp``.
    """
    if os.path.isdir("/testbed"):
        return "/testbed"
    return "/tmp"


def _validate_path(path: str) -> str:
    """Resolve and validate that *path* is within the working directory.

    Joins *path* against the sandbox root before resolution, so that an
    absolute path like ``/etc/passwd`` is anchored under the sandbox and
    rejected rather than escaping.

    Returns the resolved absolute path on success, or raises ``ValueError``
    if the path escapes the sandbox.
    """
    cwd = os.path.realpath(_resolve_cwd())
    resolved = os.path.realpath(os.path.join(cwd, path))
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        raise ValueError(
            f"Path {path!r} resolves to {resolved!r}, which is outside {cwd!r}"
        )
    return resolved


def _run_bash(command: str, timeout: int = 300) -> str:
    """Execute a bash command and return stdout + stderr.

    Note: uses ``shell=True`` intentionally — the LLM needs to execute
    arbitrary bash with pipes, redirects, and chaining. Security is provided
    by the sandbox container boundary, not by input sanitization.
    """
    logger.info("bash: %s", command[:200])
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_resolve_cwd(),
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"


def _run_str_replace_editor(args: dict) -> str:
    """Minimal str_replace_editor using shell commands.

    All file paths are validated to stay within the sandbox working directory.
    """
    command = args.get("command", "")
    path = args.get("path", "")
    file_text = args.get("file_text", "")
    view_range = args.get("view_range", [])
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")

    if command == "view":
        if not path:
            return "Error: path required for view"
        try:
            safe_path = _validate_path(path)
            with open(safe_path) as f:
                lines = f.readlines()
            if view_range and len(view_range) == 2:
                start, end = max(1, view_range[0]) - 1, view_range[1]
                lines = lines[start:end]
            return "".join(
                [f"{i + 1:4d}|{line}" for i, line in enumerate(lines)]
            )
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"

    if command == "create":
        if not path:
            return "Error: path required for create"
        try:
            safe_path = _validate_path(path)
            os.makedirs(os.path.dirname(safe_path) or ".", exist_ok=True)
            with open(safe_path, "w") as f:
                f.write(file_text)
            return f"Created {path}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating {path}: {e}"

    if command == "str_replace":
        if not path or not old_str:
            return "Error: path and old_str required"
        try:
            safe_path = _validate_path(path)
            with open(safe_path) as f:
                content = f.read()
        except ValueError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        if old_str not in content:
            return (
                f"Error: old_str not found in {path}. "
                f"File content ({len(content)} chars):\n{content[:2000]}"
            )
        new_content = content.replace(old_str, new_str, 1)
        with open(safe_path, "w") as f:
            f.write(new_content)
        return f"Replaced in {path}"

    if command == "insert":
        if not path or not new_str:
            return "Error: path and new_str required"
        insert_line = args.get("insert_line", 0)
        try:
            safe_path = _validate_path(path)
            with open(safe_path) as f:
                lines = f.readlines()
        except ValueError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            lines = []
        if insert_line <= 0 or insert_line > len(lines) + 1:
            insert_line = len(lines) + 1
        lines.insert(insert_line - 1, new_str + "\n")
        with open(safe_path, "w") as f:
            f.writelines(lines)
        return f"Inserted at line {insert_line} in {path}"

    return f"Error: unknown str_replace_editor command '{command}'"


# ── Gateway API client ──────────────────────────────────────────────────────


def chat_completions(
    base_url: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    api_key: str = "",
) -> dict:
    """Call the Gateway's OpenAI-compatible /v1/chat/completions endpoint."""
    url = f"{base_url}/chat/completions"
    body = {
        "model": os.environ.get("HERMES_MODEL", "default"),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.error("Gateway HTTP %s: %s", e.code, e.read()[:500])
        raise


# ── Main loop ───────────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build a minimal system prompt for SWE-bench tasks."""
    return (
        "You are a software engineering agent fixing issues in a codebase "
        "located at /testbed.\n\n"
        "You have access to the following tools:\n"
        "- execute_bash: Run bash commands in the sandbox.\n"
        "- str_replace_editor: View, create, and edit files.\n"
        "- submit: Submit your final answer (with a summary of changes made).\n\n"
        "Use the Hermes tool-call format to invoke tools:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {<args_dict>}}\n'
        "</tool_call>\n\n"
        "After each tool call, you will receive the tool output. "
        "Continue until the task is complete, then call submit."
    )


def run_agent(
    base_url: str,
    task: str,
    tools: list[dict],
    max_turns: int = 100,
    api_key: str = "",
) -> None:
    """Run the full Hermes agent loop against the Gateway."""
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task},
    ]

    for turn in range(max_turns):
        logger.info("Turn %d/%d — calling Gateway", turn + 1, max_turns)
        response = chat_completions(base_url, messages, tools, api_key=api_key)
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "stop")
        msg = choice.get("message", {})

        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        # If Gateway decoded tool calls directly (OpenAI format)
        if not tool_calls and _HERMES_START in content:
            content, tool_calls = parse_hermes_tool_calls(content, tools)

        if tool_calls:
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                observation = execute_tool(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation,
                    }
                )
                if name in FINISH_TOOLS:
                    logger.info("Agent finished via %s at turn %d", name, turn + 1)
                    return
        else:
            # No tool calls — agent is done
            messages.append({"role": "assistant", "content": content})
            logger.info("Agent stopped (finish_reason=%s) at turn %d", finish_reason, turn + 1)
            return

    logger.warning("Agent reached max_turns=%d without finishing", max_turns)


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    task = os.environ.get("HERMES_TASK", "")
    if not task:
        logger.error("HERMES_TASK environment variable is required")
        sys.exit(1)

    base_url = os.environ.get("HERMES_BASE_URL", "")
    if not base_url:
        logger.error("HERMES_BASE_URL environment variable is required")
        sys.exit(1)

    max_turns = int(os.environ.get("AGENT_MAX_TURNS", "100"))
    model = os.environ.get("HERMES_MODEL", "default")

    # Unset proxy vars so sandbox-internal tunnel is not bypassed
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(var, None)

    logger.info("Hermes agent starting: model=%s, max_turns=%d, base_url=%s", model, max_turns, base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command in the sandbox",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to run"}
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "View, create, and edit files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert"],
                        },
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "view_range": {"type": "array", "items": {"type": "integer"}},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit the final answer after completing the task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Summary of changes made"}
                    },
                    "required": ["message"],
                },
            },
        },
    ]

    started_at = time.time()
    try:
        run_agent(base_url, task, tools, max_turns=max_turns)
    except Exception as exc:
        logger.error("Agent loop failed: %s", exc, exc_info=True)
        sys.exit(1)
    elapsed = time.time() - started_at
    logger.info("Hermes agent finished in %.1fs", elapsed)


if __name__ == "__main__":
    main()
