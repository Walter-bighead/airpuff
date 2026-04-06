#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
SKETCH_ROOT="${2:-${HOME}/airpuff_esp32/esp32_stub}"
ARDUINO_CLI="${HOME}/.local/bin/arduino-cli"
FQBN="${FQBN:-esp32:esp32:esp32s3}"

if [[ ! -x "${ARDUINO_CLI}" ]]; then
  echo "arduino-cli not found at ${ARDUINO_CLI}" >&2
  exit 1
fi

if [[ ! -f "${SKETCH_ROOT}/esp32_stub.ino" ]]; then
  echo "sketch not found: ${SKETCH_ROOT}/esp32_stub.ino" >&2
  exit 1
fi

"${ARDUINO_CLI}" compile --fqbn "${FQBN}" "${SKETCH_ROOT}"
"${ARDUINO_CLI}" upload -p "${PORT}" --fqbn "${FQBN}" "${SKETCH_ROOT}"
