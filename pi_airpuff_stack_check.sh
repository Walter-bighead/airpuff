#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:-http://192.168.31.240:5000/api/sense}"
SERIAL_PORT="${2:-/dev/ttyACM0}"
ITERS="${ITERS:-30}"
LOG_DIR="${AIRPUFF_TEST_LOG_DIR:-${HOME}/airpuff_logs}"

mkdir -p "${LOG_DIR}"

echo "[+] Quick health check"
python3 "${HOME}/airpuff_system_debug.py" --server "${SERVER}" --serial "${SERIAL_PORT}" --iters 5

echo "[+] Soak test"
python3 "${HOME}/airpuff_soak.py" \
  --server "${SERVER}" \
  --serial "${SERIAL_PORT}" \
  --iters "${ITERS}" \
  --log-path "${LOG_DIR}/soak_$(date +%Y%m%d_%H%M%S).jsonl" \
  --summary-path "${LOG_DIR}/soak_latest_summary.json"
