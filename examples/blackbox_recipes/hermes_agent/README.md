# Hermes Agent In-Sandbox Execution

## Overview

A self-contained Python agent with Hermes tool-call format runs inside the
SWE-bench sandbox through a sidecar tool image. The external runner creates
the sandbox, mounts the tool image at `/opt/hermes-agent`, runs the Python
entrypoint against the gateway URL, and collects raw output.

The runner is **scoring-agnostic**: it optionally runs sandbox tests (if
`tools_kwargs.scoring.sandbox_eval` is true) and posts all raw data to the
Gateway. The actual scoring is handled by a pluggable `custom_reward_function`
(verl pattern), e.g. `uni_agent.reward.llm_judge.compute_score`.

The entrypoint (`run_hermes_in_sandbox.py`) uses **Python stdlib only** (no
pip dependencies) — it calls the Gateway's OpenAI-compatible
`/v1/chat/completions` endpoint, parses Hermes-format tool calls
(`<tool_call>{"name": "...", "arguments": {...}}</tool_call>`), and executes
tools via subprocess inside the sandbox.

Unlike the Claude Code recipe (which uses `claude -p ...`), this recipe uses a
Python entrypoint script. The Gateway handles Hermes tool-call parsing
(`tool_parser: hermes` in the training config), so the Gateway decodes tool
calls into OpenAI format before reaching the entrypoint.

**This recipe is self-contained.** It shares only
[`../sandbox_client.py`](../sandbox_client.py) with the other blackbox recipes;
everything else lives in this directory.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `hermes_agent` | Hermes-format Python agent entrypoint |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `akernel_sdk.Mount` and `sandbox.commands.run()` |

## Architecture

```text
[Rollouter Host: hermes_agent_runner]
  |
  |-- SandboxClient.create(image, sidecar_image, sidecar_target="/opt/hermes-agent")
  |     `-- akernel: Sandbox(mounts=[Mount(target="/opt/hermes-agent", ...)])
  |
  |-- sandbox.run("python /opt/hermes-agent/run_hermes_in_sandbox.py")
  |     `-- [Inside Sandbox]
  |           Python stdlib HTTP calls to Gateway /v1/chat/completions
  |           Hermes format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
  |           Tool execution via subprocess in /testbed
  |
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Scoring

Per-sample scoring configuration via `tools_kwargs.scoring`:

```python
# Sandbox tests only (default)
{"sandbox_eval": True}

# LLM judge only (subjective tasks)
{"sandbox_eval": False, "llm_judge": True}

# Both: weighted combination
{"sandbox_eval": True, "llm_judge": True, "sandbox_weight": 0.5}
```

Scoring is executed by the pluggable `custom_reward_function` specified in the
training config (e.g. `uni_agent.reward.llm_judge.compute_score`). The runner
only collects raw data — no scoring decisions are made in the runner.

## Prerequisites

1. **AKernel** — set `AKERNEL_SERVER_ADDRESS` and `AKERNEL_TOKEN`.
2. **Tool image** — build the hermes-agent tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

The tool image is minimal — just `python:3.11-slim` with the entrypoint script
copied to `/opt/hermes-agent/`.

```bash
# Build locally
bash examples/blackbox_recipes/hermes_agent/build_tool.sh

# Build and push to a remote registry
bash examples/blackbox_recipes/hermes_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

## 2. Training (Fully Async)

```bash
AKERNEL_SERVER_ADDRESS="6.2.179.37:8888" \
AKERNEL_TOKEN="<token>" \
HERMES_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/hermes-agent-tool:latest \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/blackbox_recipes/hermes_agent/run_train.sh
```

The training YAML keeps `hermes_agent` as the only runner:

```yaml
agent_runners:
  hermes_agent:
    runner_fqn: examples.blackbox_recipes.hermes_agent.hermes_agent_runner.hermes_agent_runner
```

The Gateway's `tool_parser_name` is set to `hermes` via
`actor_rollout_ref.rollout.multi_turn.format: hermes`, so tool calls are
decoded server-side.

## 3. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_TURNS` | `100` | Max conversation turns; read by the entrypoint |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout (seconds) |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Max wall time for the agent process |
| `HERMES_AGENT_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/hermes-agent-tool:latest` | Sidecar tool image |
| `CONDA_ENV` | `testbed` | Conda env activated before running the entrypoint |

## Differences from the Claude Code Recipe

| Aspect | Claude Code | Hermes Agent |
|--------|------------|--------------|
| Agent binary | `claude` (npm package) | Python script (stdlib only) |
| API format | Anthropic Messages (`/v1/messages`) | OpenAI Chat Completions (`/v1/chat/completions`) |
| Tool format | Claude Code native | Hermes `<tool_call>{"name": ..., "arguments": ...}</tool_call>` |
| Gateway parser | Anthropic adapter (`anthropic_to_internal`) | Hermes parser (`HermesToolParser`) |
| Sidecar image | `FROM scratch` with npm-built claude | `python:3.11-slim` with entrypoint script |
| In-sandbox invocation | `claude -p <task>` | `python /opt/hermes-agent/run_hermes_in_sandbox.py` |
| Dependencies | Node.js (bundled in sidecar) | Python stdlib (already in sandbox) |
