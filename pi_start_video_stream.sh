#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-720p}"
case "$PROFILE" in
  540|540p)
    export STREAM_WIDTH=960
    export STREAM_HEIGHT=540
    export CAM_FPS=24
    export UPLOAD_FPS=10
    export MJPEG_QUALITY=80
    ;;
  720|720p)
    export STREAM_WIDTH=1280
    export STREAM_HEIGHT=720
    export CAM_FPS=20
    export UPLOAD_FPS=8
    export MJPEG_QUALITY=82
    ;;
  *)
    echo "Usage: $0 [540p|720p]"
    exit 1
    ;;
esac

pkill -f /tmp/airpuff_rpicam_stream.py || true
pkill -f /tmp/airpuff_mjpeg_push.py || true
pkill -f pi_video_stream_push.py || true
pkill -f pi_rpicam_stream_server.sh || true
pkill -f '/tmp/rpicam_8554_supervisor.sh' || true
pkill -f 'rpicam-vid.*8554' || true
pkill -f "rpicam-vid -n --codec mjpeg --width" || true

: > /tmp/airpuff_mjpeg_push.log

nohup env \
  STREAM_WIDTH="$STREAM_WIDTH" \
  STREAM_HEIGHT="$STREAM_HEIGHT" \
  CAM_FPS="$CAM_FPS" \
  UPLOAD_FPS="$UPLOAD_FPS" \
  MJPEG_QUALITY="$MJPEG_QUALITY" \
  python3 ~/airpuff/pi_video_stream_push.py >/tmp/airpuff_mjpeg_push.log 2>&1 &

echo "video stream started profile=$PROFILE res=${STREAM_WIDTH}x${STREAM_HEIGHT} cam_fps=${CAM_FPS} upload_fps=${UPLOAD_FPS}"
ps -ef | grep -E 'pi_video_stream_push.py|rpicam-vid -n --codec mjpeg --quality' | grep -v grep
