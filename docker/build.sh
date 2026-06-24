#!/usr/bin/env bash
# 构建 hermes-eval 镜像。
# 上游 hermes-agent 不在本仓库里，首次会从 GitHub 浅克隆到一个临时目录后构建。
set -euo pipefail

cd "$(dirname "$0")/.."

HERMES_IMAGE="${HERMES_IMAGE:-hermes-agent:latest}"
EVAL_IMAGE="${EVAL_IMAGE:-hermes-eval:latest}"
HERMES_REPO="${HERMES_REPO:-https://github.com/nousresearch/hermes-agent}"
HERMES_REF="${HERMES_REF:-v2026.5.16}"

if ! docker image inspect "$HERMES_IMAGE" >/dev/null 2>&1; then
    workdir="$(mktemp -d)"
    trap 'rm -rf "$workdir"' EXIT
    echo "Base image $HERMES_IMAGE not found; cloning $HERMES_REPO@$HERMES_REF -> $workdir"
    git clone --depth 1 --branch "$HERMES_REF" "$HERMES_REPO" "$workdir/hermes-agent"
    docker build -t "$HERMES_IMAGE" "$workdir/hermes-agent"
fi

echo "Building $EVAL_IMAGE on top of $HERMES_IMAGE ..."
docker build \
    --build-arg "BASE=$HERMES_IMAGE" \
    -t "$EVAL_IMAGE" \
    -f docker/Dockerfile.eval \
    docker/

echo "Done. Use: $EVAL_IMAGE"
