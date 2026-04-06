#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_FLAGS=(--user --break-system-packages --retries 5 --timeout 60)

install_with_retry() {
  local label="$1"
  shift
  local attempt
  for attempt in 1 2 3; do
    echo "[+] Installing ${label} (attempt ${attempt}/3)"
    if "${PYTHON_BIN}" -m pip install "${PIP_FLAGS[@]}" "$@"; then
      return 0
    fi
    sleep 2
  done
  echo "[-] Failed to install ${label}" >&2
  return 1
}

install_with_retry "base" flask requests
install_with_retry "numpy" numpy
install_with_retry "opencv" opencv-python-headless
install_with_retry "faster-whisper" faster-whisper

echo "[+] Running environment check"
"${PYTHON_BIN}" "${HOME}/laptop_check_airpuff_env.py"
