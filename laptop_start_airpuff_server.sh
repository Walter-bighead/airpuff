#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-minimal}"
APP_PATH="${2:-${HOME}/airpuff_server.py}"
HOST="${AIRPUFF_HOST:-0.0.0.0}"
PORT="${AIRPUFF_PORT:-5000}"
STDOUT_LOG="${AIRPUFF_SERVER_STDOUT_LOG:-${HOME}/airpuff_server.log}"
JSONL_LOG="${AIRPUFF_LOG_PATH:-}"

if [[ ! -f "${APP_PATH}" ]]; then
  echo "server file not found: ${APP_PATH}" >&2
  exit 1
fi

OLD_PIDS="$(pgrep -f "python3 .*airpuff_server.py" || true)"
if [[ -n "${OLD_PIDS}" ]]; then
  echo "[+] Stopping old server pid(s): ${OLD_PIDS}"
  kill ${OLD_PIDS}
  sleep 1
fi

COMMON_ENV=(
  "AIRPUFF_HOST=${HOST}"
  "AIRPUFF_PORT=${PORT}"
)

if [[ -n "${JSONL_LOG}" ]]; then
  COMMON_ENV+=("AIRPUFF_LOG_PATH=${JSONL_LOG}")
fi

case "${PROFILE}" in
  minimal)
    PROFILE_ENV=(
      "AIRPUFF_ENABLE_LLM=0"
      "AIRPUFF_ENABLE_VLM=0"
      "AIRPUFF_ENABLE_WHISPER=0"
      "AIRPUFF_VISION_MODE=off"
    )
    ;;
  vision-lite)
    PROFILE_ENV=(
      "AIRPUFF_ENABLE_LLM=0"
      "AIRPUFF_ENABLE_VLM=0"
      "AIRPUFF_ENABLE_WHISPER=0"
      "AIRPUFF_VISION_MODE=lite"
    )
    ;;
  vision-flow)
    PROFILE_ENV=(
      "AIRPUFF_ENABLE_LLM=0"
      "AIRPUFF_ENABLE_VLM=0"
      "AIRPUFF_ENABLE_WHISPER=0"
      "AIRPUFF_VISION_MODE=flow"
    )
    ;;
  full)
    PROFILE_ENV=(
      "AIRPUFF_CMD_FAST_PATH=1"
      "AIRPUFF_ENABLE_VLM=0"
      "AIRPUFF_VISION_MODE=${AIRPUFF_VISION_MODE:-flow}"
    )
    ;;
  full-vlm)
    PROFILE_ENV=(
      "AIRPUFF_CMD_FAST_PATH=1"
      "AIRPUFF_VISION_MODE=vlm"
    )
    ;;
  *)
    echo "unknown profile: ${PROFILE}" >&2
    echo "use: minimal | vision-lite | vision-flow | full | full-vlm" >&2
    exit 1
    ;;
esac

echo "[+] Starting AirPuff server profile=${PROFILE} port=${PORT}"
nohup env "${COMMON_ENV[@]}" "${PROFILE_ENV[@]}" python3 "${APP_PATH}" > "${STDOUT_LOG}" 2>&1 &
sleep 2

NEW_PID="$(pgrep -f "python3 ${APP_PATH}" | tail -n 1 || true)"
if [[ -z "${NEW_PID}" ]]; then
  echo "server failed to start, see ${STDOUT_LOG}" >&2
  exit 1
fi

echo "[+] AirPuff server pid=${NEW_PID}"
tail -n 20 "${STDOUT_LOG}"
