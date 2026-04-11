#!/usr/bin/env bash
set -euo pipefail

# Start AirPuff server with a trained YOLO model and monocular geometry distance mode.
#
# Usage:
#   ./laptop_start_trained_yolo.sh [model_path] [profile]
# Example:
#   ./laptop_start_trained_yolo.sh /home/walter/airpuff/models/best.pt full

MODEL_PATH="${1:-${HOME}/airpuff/models/best.pt}"
PROFILE="${2:-full}"
APP_PATH="${HOME}/airpuff_server.py"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] model not found: ${MODEL_PATH}" >&2
  exit 1
fi

export AIRPUFF_VISION_MODE=yolo
export AIRPUFF_YOLO_MODEL="${MODEL_PATH}"
export AIRPUFF_YOLO_DISTANCE_MODE="${AIRPUFF_YOLO_DISTANCE_MODE:-ground_plane}"
export AIRPUFF_YOLO_CAMERA_HEIGHT_M="${AIRPUFF_YOLO_CAMERA_HEIGHT_M:-0.935}"
export AIRPUFF_YOLO_CAMERA_PITCH_DEG="${AIRPUFF_YOLO_CAMERA_PITCH_DEG:-5.16}"

echo "[start] profile=${PROFILE}"
echo "[model] ${AIRPUFF_YOLO_MODEL}"
echo "[distance_mode] ${AIRPUFF_YOLO_DISTANCE_MODE}"
echo "[camera_height_m] ${AIRPUFF_YOLO_CAMERA_HEIGHT_M}"
echo "[camera_pitch_deg] ${AIRPUFF_YOLO_CAMERA_PITCH_DEG}"

"${HOME}/laptop_start_airpuff_server.sh" "${PROFILE}" "${APP_PATH}"
echo "[url] http://10.42.0.1:5000/"
