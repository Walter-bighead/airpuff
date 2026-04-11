#!/usr/bin/env bash
set -euo pipefail

# Configure Raspberry Pi network + stream target for AirPuff AP mode.
# Usage:
#   ./pi_configure_airpuff_ap.sh [laptop_ip] [pi_ip] [ssid]
# Example:
#   ./pi_configure_airpuff_ap.sh 10.42.0.1 10.42.0.82 AirPuff_AP

LAPTOP_IP="${1:-10.42.0.1}"
PI_IP="${2:-10.42.0.82}"
AP_SSID="${3:-AirPuff_AP}"
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(getent passwd "${USER_NAME}" | cut -d: -f6)"
PROJECT_DIR="${HOME_DIR}/airpuff"
SYSTEMD_SERVICE_NAME="pi_airpuff_video_stream.service"
SYSTEMD_SERVICE_PATH="/etc/systemd/system/${SYSTEMD_SERVICE_NAME}"
SERVICE_TEMPLATE="${PROJECT_DIR}/pi_airpuff_video_stream.service"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo bash "$0" "$@"
fi

echo "[info] AirPuff AP profile setup"
echo "[info] laptop_ip=${LAPTOP_IP} pi_ip=${PI_IP} ssid=${AP_SSID}"

if ! command -v nmcli >/dev/null 2>&1; then
  echo "[error] nmcli not found. Install NetworkManager first." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found." >&2
  exit 1
fi

CONN_NAME="$(nmcli -t -f NAME,TYPE connection show | awk -F: -v ssid="${AP_SSID}" '$2=="802-11-wireless" && $1==ssid {print $1; exit}')"
if [[ -z "${CONN_NAME}" ]]; then
  echo "[error] Wi-Fi profile '${AP_SSID}' not found. Connect once manually, then rerun." >&2
  exit 1
fi

echo "[step] Configure static IPv4 for ${CONN_NAME}"
nmcli connection modify "${CONN_NAME}" \
  ipv4.method manual \
  ipv4.addresses "${PI_IP}/24" \
  ipv4.gateway "${LAPTOP_IP}" \
  ipv4.dns "${LAPTOP_IP}" \
  ipv6.method auto

if [[ ! -f "${SYSTEMD_SERVICE_PATH}" ]]; then
  if [[ -f "${SERVICE_TEMPLATE}" ]]; then
    echo "[step] Install service template to ${SYSTEMD_SERVICE_PATH}"
    cp -f "${SERVICE_TEMPLATE}" "${SYSTEMD_SERVICE_PATH}"
  else
    echo "[error] Service file missing: ${SYSTEMD_SERVICE_PATH} and template not found at ${SERVICE_TEMPLATE}" >&2
    exit 1
  fi
fi

echo "[step] Set stream target to control page address"
sed -i.bak -E "s|^Environment=AIRPUFF_SENSE_URL=.*$|Environment=AIRPUFF_SENSE_URL=http://${LAPTOP_IP}:5000/api/sense|" "${SYSTEMD_SERVICE_PATH}"

echo "[step] Reload + restart service"
systemctl daemon-reload
systemctl enable "${SYSTEMD_SERVICE_NAME}" >/dev/null 2>&1 || true
systemctl restart "${SYSTEMD_SERVICE_NAME}"

echo "[step] Reconnect Wi-Fi profile ${CONN_NAME}"
nmcli connection up "${CONN_NAME}" >/dev/null 2>&1 || true

echo "[ok] Done"
echo "[check] ip -4 addr show wlan0 | grep -E 'inet '"
echo "[check] systemctl status ${SYSTEMD_SERVICE_NAME} --no-pager"
echo "[check] curl -s http://${LAPTOP_IP}:5000/api/health"
