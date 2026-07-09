# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Uni-Agent is a unified framework for building, running, and RL-training general-purpose agents at scale. It supports 1000+ concurrent agent tasks with the same interaction stack used for both inference and GRPO/GSPO-style RL training. The project lives under the `verl-project` GitHub org and depends on `verl` (git submodule) as its RL training engine.

## Commands

### Setup

```bash
git submodule update --init --recursive
pip install --no-deps -e ./verl
pip install swe-rex loguru pydantic pydantic_settings aiohttp
pip install -e .
```

### Lint & Type Check

```bash
pre-commit install
pre-commit run --all-files    # runs ruff, ruff-format, mypy, compileall
```

The `verl/` submodule is excluded from all checks. `uni_agent/tools/` is excluded from some checks (compileall). Ruff line length is 120 (not the standard 88 — this project uses its own setting). Ruff rules: `E`, `F`, `UP`, `B`, `I`, `G`.

### Test

Tests use pytest. No special config file — standard pytest discovery under `tests/`:

```bash
pytest                          # all tests
pytest tests/uni_agent/gateway/ # single package
pytest -n auto                  # parallel (if pytest-xdist installed)
```

Test directories mirror source structure:
- `tests/uni_agent/gateway/` — GatewayActor, GatewayManager
- `tests/uni_agent/framework/` — training framework (generate_sequences, multi-modal postprocess)
- `tests/uni_agent/deployment/` — host runtime
- `tests/deployment/` — local, modal deployments
- `tests/interaction/` — logging utilities

### Run Dashboard

```bash
python -m dashboard.server --log-dir /tmp/swebench_qwen3_coder --port 8765
```

## Architecture

### Core Abstraction: Model-Tool-Env Triad

The framework is organized around three independently swappable layers:

- **Model** — the reasoning backend (LLM) that decides what to do next
- **Tool** — how the model perceives and acts on the environment (tool definitions, schemas, parsing)
- **Env** — the runtime sandbox where actions execute (SWE-ReX-based containers/VMs)

These compose into `AgentInteraction` (single model-call + tool-execution cycle), orchestrated by `UniAgentLoop` (extends verl's `AgentLoopBase`).

### Key Packages

| Package | Role |
|---------|------|
| `uni_agent/interaction/` | Core interaction loop: `AgentInteraction`, `AgentEnv`, `AgentChatModel`/`OpenAICompatibleChatModel`, `ToolsManager`, `ToolParser` |
| `uni_agent/gateway/` | OpenAI-compatible HTTP gateway between agent runners and LLM backends. Captures token-level trajectories (prompt_ids, response_ids, logprobs) for RL training while exposing standard `/v1/chat/completions`. Uses Ray actors + FastAPI. |
| `uni_agent/framework/` | Training framework integration with verl. `AgentFramework` (ABC), `OpenAICompatibleAgentFramework`, `AgentFrameworkRolloutAdapter`, `AgentFrameworkWorker` (Ray actor). |
| `uni_agent/agent_loop.py` | `UniAgentLoop` — the main loop: env startup, tool/skill install, interaction loop, reward computation, trajectory output to verl's `TransferQueue`. |
| `uni_agent/tools/` | Built-in tool definitions with OpenAI-compatible schemas. Decorator-based registry (`register_tool`/`get_tool`). |
| `uni_agent/skills/` | Progressive-disclosure skill system (SKILL.md + sibling files, manifest in system prompt, lazy body reads). |
| `uni_agent/deployment/` | Six pluggable backends: `host`, `local` (Docker/Apptainer), `local_attach`, `local_native` (pexpect), `modal`, `vefaas`. All via SWE-ReX `AbstractRuntime`. |
| `uni_agent/reward/` | Reward spec registry for RL training (`swe_bench`, `swe_rebench`, `r2e_gym`, `search`, `terminal_bench`, etc.). |
| `uni_agent/async_logging.py` | Loguru-based async logging with per-run-id file routing. |

### Dual-Path Model Backends

The same `AgentInteraction` loop works with two model clients:
- **`AgentChatModel`** — training path: token-level interface (`prompt_ids`/`response_ids`) via verl's LLM server
- **`OpenAICompatibleChatModel`** — inference path: standard `/v1/chat/completions` API, string-level interface

### Gateway Pattern (Training)

During RL training, `GatewayManager` (driver-side) manages a pool of `GatewayActor`s (Ray remote actors, each running an in-process FastAPI server). The actors provide an OpenAI-compatible HTTP API to agent runners, while `GatewaySession` + `MessageCodec` capture the token-level data (logprobs, response_ids) needed for RL loss computation. This decouples agent runners from the LLM backend.

### Key Patterns

- **Registry + lazy loading:** Tools (`register_tool`) and reward specs (`register_reward_spec`) use decorator-based registries. Modules are imported lazily to avoid loading unused dependencies.
- **`auto_await` decorator:** Handles sync/async transparency — deployment methods can be written as sync or async; the decorator adapts at call time.
- **Ray-based distribution:** Gateway actors use Ray with node-affinity scheduling. Agent runners support `inline_async` (in-process) and `ray_task` (distributed) dispatch modes, with `asyncio.Semaphore` for concurrency control.
- **verl submodule:** Version is read from `verl/version/version`. The submodule is pinned — treat it as read-only unless explicitly updating it.

## Project Conventions

- Line length: 120 (ruff), not 88
- Type hints: used throughout with Pydantic v2 for config/schema validation
- Config system: OmegaConf (YAML), hierarchical — follows verl's config patterns
- Logging: loguru with `async_logging.py` routing; never use `print()` for log output
- The `verl/` submodule is pinned — do not modify it without explicit instruction
- GitHub Actions CI runs pre-commit, docs build, PR title validation, and secret scanning
