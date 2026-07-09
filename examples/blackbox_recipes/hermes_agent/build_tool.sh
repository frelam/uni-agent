#!/usr/bin/env bash
# Build the Hermes Agent sidecar tool image.
#
# The image copies run_hermes_in_sandbox.py into a minimal Python image.
# It is mounted into the SWE-bench sandbox at /opt/hermes-agent, and executed
# with the sandbox's own Python interpreter.
#
# Usage:
#   bash examples/blackbox_recipes/hermes_agent/build_tool.sh
#   bash examples/blackbox_recipes/hermes_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-hermes-agent-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"

# Parse args
REGISTRY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Building hermes-agent tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.hermes-agent-tool" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${SCRIPT_DIR}/"

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Tagging and pushing: ${FULL_TAG}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"
    docker push "${FULL_TAG}"
    echo "    Pushed."
fi

echo ""
echo "Tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
    echo "  Remote sandbox: ${FULL_TAG}"
fi
