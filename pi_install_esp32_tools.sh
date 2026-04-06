#!/usr/bin/env bash
set -euo pipefail

ARDUINO_CLI_VERSION="${ARDUINO_CLI_VERSION:-1.4.1}"
ARDUINO_BASE="${HOME}/.local/share/arduino"
ARDUINO_BIN_DIR="${HOME}/.local/bin"
ARDUINO_CLI="${ARDUINO_BIN_DIR}/arduino-cli"
ARDUINO_CONFIG="${HOME}/.arduino15/arduino-cli.yaml"
ARDUINO_DIR_USER="${HOME}/Arduino"
ESP32_INDEX_URL="https://espressif.github.io/arduino-esp32/package_esp32_index.json"
ESPTOOL_VENV="${HOME}/.venvs/esptool"

mkdir -p "${ARDUINO_BIN_DIR}" "${ARDUINO_BASE}" "${HOME}/.venvs" "${HOME}/.arduino15" "${ARDUINO_DIR_USER}"

if [[ ! -x "${ARDUINO_CLI}" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  curl -L "https://downloads.arduino.cc/arduino-cli/arduino-cli_${ARDUINO_CLI_VERSION}_Linux_ARM64.tar.gz" -o "${tmpdir}/arduino-cli.tar.gz"
  tar -xzf "${tmpdir}/arduino-cli.tar.gz" -C "${tmpdir}"
  install -m 0755 "${tmpdir}/arduino-cli" "${ARDUINO_CLI}"
fi

cat > "${ARDUINO_CONFIG}" <<EOF
board_manager:
  additional_urls:
    - ${ESP32_INDEX_URL}
directories:
  data: ${HOME}/.arduino15
  downloads: ${HOME}/.arduino15/staging
  user: ${ARDUINO_DIR_USER}
library:
  enable_unsafe_install: false
sketch:
  always_export_binaries: false
updater:
  enable_notification: false
EOF

"${ARDUINO_CLI}" core update-index
"${ARDUINO_CLI}" core install esp32:esp32

if [[ ! -d "${ESPTOOL_VENV}" ]]; then
  python3 -m venv "${ESPTOOL_VENV}"
fi
"${ESPTOOL_VENV}/bin/pip" install --upgrade pip
"${ESPTOOL_VENV}/bin/pip" install esptool

echo "arduino-cli: ${ARDUINO_CLI}"
echo "esptool: ${ESPTOOL_VENV}/bin/esptool.py"
