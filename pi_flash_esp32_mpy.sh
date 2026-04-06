#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
FW_URL="${FW_URL:-https://micropython.org/resources/firmware/ESP32_GENERIC_S3-SPIRAM_OCT-20251209-v1.27.0.bin}"
FW_DIR="${HOME}/airpuff_esp32/firmware"
FW_PATH="${FW_DIR}/$(basename "${FW_URL}")"
ESPTOOL="${HOME}/.local/bin/esptool"

mkdir -p "${FW_DIR}"

if [[ ! -x "${ESPTOOL}" ]]; then
  echo "esptool not found at ${ESPTOOL}" >&2
  echo "Install with: python3 -m pip install --user --break-system-packages esptool" >&2
  exit 1
fi

if [[ ! -f "${FW_PATH}" ]]; then
  echo "[+] Downloading ${FW_URL}"
  curl -L --fail "${FW_URL}" -o "${FW_PATH}"
fi

echo "[+] Flashing ${FW_PATH} to ${PORT}"
"${ESPTOOL}" --chip esp32s3 --port "${PORT}" erase-flash
"${ESPTOOL}" --chip esp32s3 --port "${PORT}" --baud 460800 write-flash -z 0 "${FW_PATH}"

echo "[+] MicroPython firmware flashed."
echo "[+] Next step: ./pi_upload_esp32_mpy.sh ${PORT}"
