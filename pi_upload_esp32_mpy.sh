#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
APP_PATH="${2:-${HOME}/esp32_stub_mpy.py}"
MPREMOTE="${HOME}/.local/bin/mpremote"
CUSTOM_PUSH="${HOME}/esp32_mpy_push.py"

if [[ ! -f "${APP_PATH}" ]]; then
  echo "MicroPython app not found: ${APP_PATH}" >&2
  exit 1
fi

if [[ -f "${CUSTOM_PUSH}" ]]; then
  echo "[+] Uploading ${APP_PATH} to ${PORT} as main.py via custom uploader"
  python3 "${CUSTOM_PUSH}" --port "${PORT}" --src "${APP_PATH}" --dst main.py
  echo "[+] main.py uploaded and board soft-reset."
  exit 0
fi

if [[ ! -x "${MPREMOTE}" ]]; then
  echo "[+] Installing mpremote"
  python3 -m pip install --user --break-system-packages mpremote
fi

echo "[+] Uploading ${APP_PATH} to ${PORT} as main.py via mpremote"
"${MPREMOTE}" connect "${PORT}" fs cp "${APP_PATH}" :main.py
"${MPREMOTE}" connect "${PORT}" reset

echo "[+] main.py uploaded and board reset."
