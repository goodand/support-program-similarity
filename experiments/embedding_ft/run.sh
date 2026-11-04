#!/usr/bin/env bash
# Usage: EMBED_FT_WORK=<work dir> MODEL_DIR=<dir holding Qwen__Qwen3-Embedding-8B> bash run.sh [script.py under code/] [script args...]
# EMBED_FT_WORK gets data/ (read-only), output/ and logs/; nothing is written inside this repo.
# Optional: INPUT_DIR=<dir> is mounted read-only at /input; IMAGE=<image> (default embed-ft:20261102-v1, vLLM: vllm/vllm-openai:v0.11.0).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))  # Git Bash: D:/ form for docker.exe
: "${EMBED_FT_WORK:?set EMBED_FT_WORK}" "${MODEL_DIR:?set MODEL_DIR}"
IMAGE=${IMAGE:-embed-ft:20261102-v1}
SCRIPT=${1:-smoke_qwen_lora.py}
[ $# -gt 0 ] && shift
mkdir -p "$EMBED_FT_WORK/data" "$EMBED_FT_WORK/output" "$EMBED_FT_WORK/logs"
EXTRA=()
[ -n "${INPUT_DIR:-}" ] && EXTRA+=(-v "$INPUT_DIR:/input:ro")
NAME=embed-ft-${SCRIPT%.py}-$(date +%Y%m%d-%H%M%S)-v1
MSYS_NO_PATHCONV=1 docker run --rm --name "$NAME" --gpus all --shm-size=16g --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -v "$MODEL_DIR:/models:ro" "${EXTRA[@]}" \
  -v "$HERE/code:/code:ro" -v "$EMBED_FT_WORK/data:/data:ro" -v "$EMBED_FT_WORK/output:/output" \
  -w /code "$IMAGE" /code/"$SCRIPT" "$@" 2>&1 | tee "$EMBED_FT_WORK/logs/${NAME}.log"
