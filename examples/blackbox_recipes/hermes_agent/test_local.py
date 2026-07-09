"""Local smoke test for the Hermes agent entrypoint.

Tests the entrypoint with a mock Gateway HTTP server — no sandbox, no GPU,
no Ray needed. Just Python stdlib.

Usage:
    python examples/blackbox_recipes/hermes_agent/test_local.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


# ── Mock Gateway server ──────────────────────────────────────────────────────

class MockGatewayHandler(BaseHTTPRequestHandler):
    """Simulates a Gateway that returns Hermes-format tool calls."""

    turn = 0
    responses = [
        # Turn 1: execute_bash to list files
        {
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": 'Let me look at the files.\n<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls /tmp"}}\n</tool_call>',
                },
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
        # Turn 2: execute_bash to check Python version
        {
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": 'Let me check the Python version.\n<tool_call>\n{"name": "execute_bash", "arguments": {"command": "python --version"}}\n</tool_call>',
                },
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
        },
        # Turn 3: submit
        {
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": 'Done. Let me submit.\n<tool_call>\n{"name": "submit", "arguments": {"message": "All good"}}\n</tool_call>',
                },
            }],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
        },
    ]

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

        # Log the request
        messages = body.get("messages", [])
        last_msg = messages[-1] if messages else {}
        print(f"  [MockGateway] turn={self.__class__.turn}, last_role={last_msg.get('role', '?')}, "
              f"content_preview={str(last_msg.get('content', ''))[:80]}")

        if self.__class__.turn >= len(self.__class__.responses):
            response = {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Done."},
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        else:
            response = self.__class__.responses[self.__class__.turn]
        self.__class__.turn += 1

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs


def start_mock_gateway(port: int = 18888) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), MockGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[MockGateway] Listening on http://127.0.0.1:{port}")
    return server


# ── Test: Hermes tool call parser ────────────────────────────────────────────


def test_parse_hermes_tool_calls():
    """Verify Hermes-format parsing without a server."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from examples.blackbox_recipes.hermes_agent.run_hermes_in_sandbox import parse_hermes_tool_calls

    tools = [
        {"type": "function", "function": {"name": "execute_bash", "description": "..."}},
        {"type": "function", "function": {"name": "submit", "description": "..."}},
        {"type": "function", "function": {"name": "str_replace_editor", "description": "..."}},
    ]

    # Single tool call
    text = 'Before text\n<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>\nAfter text'
    content, tool_calls = parse_hermes_tool_calls(text, tools)
    assert content == "Before text", f"Expected 'Before text', got {content!r}"
    assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
    assert tool_calls[0]["function"]["name"] == "execute_bash"

    # Multiple tool calls
    text = '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>\n<tool_call>\n{"name": "submit", "arguments": {"message": "done"}}\n</tool_call>'
    content, tool_calls = parse_hermes_tool_calls(text, tools)
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "execute_bash"
    assert tool_calls[1]["function"]["name"] == "submit"

    # No tool calls
    text = "Just a plain response, no tools here."
    content, tool_calls = parse_hermes_tool_calls(text, tools)
    assert content == text
    assert len(tool_calls) == 0

    print("  ✓ parse_hermes_tool_calls: all assertions passed")


# ── Test: basic tool execution ───────────────────────────────────────────────


def test_execute_tool():
    """Verify tool execution functions."""
    from examples.blackbox_recipes.hermes_agent.run_hermes_in_sandbox import execute_tool

    # execute_bash
    result = execute_tool("execute_bash", json.dumps({"command": "echo hello"}))
    assert "hello" in result.lower(), f"Unexpected result: {result}"

    # submit
    result = execute_tool("submit", json.dumps({"message": "done"}))
    assert "finished" in result.lower(), f"Unexpected result: {result}"

    # Unknown tool
    result = execute_tool("unknown_tool", "{}")
    assert "unknown" in result.lower(), f"Unexpected result: {result}"

    print("  ✓ execute_tool: all assertions passed")


# ── Test: end-to-end with mock gateway ───────────────────────────────────────


def test_entrypoint_with_mock_gateway():
    """Run the entrypoint against the mock Gateway and verify it completes."""
    from examples.blackbox_recipes.hermes_agent.run_hermes_in_sandbox import run_agent

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
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
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        },
    ]

    print("  Running agent loop against mock gateway...")
    MockGatewayHandler.turn = 0  # Reset

    run_agent(
        base_url="http://127.0.0.1:18888/v1",
        task="List files in /tmp",
        tools=tools,
        max_turns=5,
    )
    print("  ✓ Agent loop completed successfully")


# ── Test: integration via subprocess ─────────────────────────────────────────


def test_entrypoint_subprocess():
    """Run the full entrypoint as a subprocess (like the runner does)."""
    entrypoint = os.path.join(
        os.path.dirname(__file__), "run_hermes_in_sandbox.py"
    )

    # Point at nothing — just test that the script starts, parses env, and fails
    # gracefully (no live gateway needed — we test the error path)
    env = {
        **os.environ,
        "HERMES_TASK": "Say hello",
        "HERMES_BASE_URL": "http://127.0.0.1:19999/v1",  # Nothing listening
        "AGENT_MAX_TURNS": "1",
        "HERMES_MODEL": "test",
        "PATH": os.environ["PATH"],
    }
    # Unset proxy vars
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        env.pop(var, None)

    result = subprocess.run(
        [sys.executable, entrypoint],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Should fail with connection refused — but NOT with a Python error
    assert result.returncode != 0, "Expected non-zero exit code (no gateway listening)"
    assert "Agent loop failed" in (result.stdout + result.stderr), \
        f"Expected 'Agent loop failed' in output, got: {result.stdout[:500]}{result.stderr[:500]}"
    print("  ✓ Subprocess invocation: graceful failure on unreachable gateway")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=== Hermes Agent Local Smoke Test ===\n")

    # Start mock gateway
    server = start_mock_gateway(18888)

    try:
        time.sleep(0.2)  # Let server start

        print("\n1. Testing Hermes tool call parser...")
        test_parse_hermes_tool_calls()

        print("\n2. Testing tool execution...")
        test_execute_tool()

        print("\n3. Testing entrypoint against mock gateway...")
        test_entrypoint_with_mock_gateway()

        print("\n4. Testing subprocess invocation...")
        test_entrypoint_subprocess()

        print("\n=== All tests passed! ===")
        return 0

    finally:
        server.shutdown()
        print("\n[MockGateway] Stopped")


if __name__ == "__main__":
    sys.exit(main())
