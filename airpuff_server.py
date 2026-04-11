import os
import math
import time
import base64
import json
import re
import pty
import select
import shlex
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string, send_file

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional runtime dependency
    WhisperModel = None
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except Exception:  # pragma: no cover - optional runtime dependency
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

app = Flask(__name__)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_csv(name: str, default: str) -> List[str]:
    value = os.getenv(name, default)
    return [item.strip().lower().replace(" ", "_") for item in value.split(",") if item.strip()]


CONFIG = {
    "OLLAMA_API": os.getenv("AIRPUFF_OLLAMA_API", "http://127.0.0.1:11434/api/generate"),
    "LLM_MODEL": os.getenv("AIRPUFF_LLM_MODEL", "lfm2:24b"),
    "LLM_CMD_MODEL": os.getenv("AIRPUFF_LLM_CMD_MODEL", ""),
    "LLM_CHAT_MODEL": os.getenv("AIRPUFF_LLM_CHAT_MODEL", ""),
    "VLM_MODEL": os.getenv("AIRPUFF_VLM_MODEL", "minicpm-v"),
    "WHISPER_MODEL": os.getenv("AIRPUFF_WHISPER_MODEL", "tiny"),
    "WHISPER_DEVICE": os.getenv("AIRPUFF_WHISPER_DEVICE", "cpu"),
    "WHISPER_COMPUTE": os.getenv("AIRPUFF_WHISPER_COMPUTE", "int8"),
    "ENABLE_LLM": env_bool("AIRPUFF_ENABLE_LLM", True),
    "ENABLE_VLM": env_bool("AIRPUFF_ENABLE_VLM", True),
    "ENABLE_WHISPER": env_bool("AIRPUFF_ENABLE_WHISPER", True),
    "CMD_FAST_PATH": env_bool("AIRPUFF_CMD_FAST_PATH", True),
    "CMD_FAST_PATH_MAX_CHARS": env_int("AIRPUFF_CMD_FAST_PATH_MAX_CHARS", 24),
    "VLM_MIN_INTERVAL_MS": env_int("AIRPUFF_VLM_MIN_INTERVAL_MS", 1200),
    "VISION_MODE": os.getenv("AIRPUFF_VISION_MODE", "vlm"),  # vlm | lite | flow | yolo | off
    "VISION_CLAHE": env_bool("AIRPUFF_VISION_CLAHE", True),
    "VISION_GAMMA": env_float("AIRPUFF_VISION_GAMMA", 1.0),
    "BRIGHT_TH": env_int("AIRPUFF_BRIGHT_TH", 240),
    "BRIGHT_FRAC_TH": env_float("AIRPUFF_BRIGHT_FRAC_TH", 0.2),
    "BRIGHT_SCALE": env_float("AIRPUFF_BRIGHT_SCALE", 2.0),
    "LITE_W": env_int("AIRPUFF_LITE_W", 160),
    "LITE_H": env_int("AIRPUFF_LITE_H", 120),
    "LITE_EDGE_TH": env_int("AIRPUFF_LITE_EDGE_TH", 20),  # percent*100
    "FLOW_W": env_int("AIRPUFF_FLOW_W", 160),
    "FLOW_H": env_int("AIRPUFF_FLOW_H", 120),
    "FLOW_MAG_TH": env_float("AIRPUFF_FLOW_MAG_TH", 2.0),
    "FLOW_MIN_INTERVAL_MS": env_int("AIRPUFF_FLOW_MIN_INTERVAL_MS", 200),
    "YOLO_MODEL": os.getenv("AIRPUFF_YOLO_MODEL", "yolo11n.pt"),
    "YOLO_IMGSZ": env_int("AIRPUFF_YOLO_IMGSZ", 512),
    "YOLO_CONF": env_float("AIRPUFF_YOLO_CONF", 0.30),
    "YOLO_MIN_INTERVAL_MS": env_int("AIRPUFF_YOLO_MIN_INTERVAL_MS", 120),
    "YOLO_MAX_DETS": env_int("AIRPUFF_YOLO_MAX_DETS", 8),
    "YOLO_BOX_HOLD_SEC": env_float("AIRPUFF_YOLO_BOX_HOLD_SEC", 0.45),
    "YOLO_MISS_TOLERANCE": env_int("AIRPUFF_YOLO_MISS_TOLERANCE", 3),
    "YOLO_OBSTACLE_DIST_M": env_float("AIRPUFF_YOLO_OBSTACLE_DIST_M", 4.2),
    "YOLO_EMERGENCY_STOP_M": env_float("AIRPUFF_YOLO_EMERGENCY_STOP_M", 2.0),
    "YOLO_HFOV_DEG": env_float("AIRPUFF_YOLO_HFOV_DEG", 62.0),
    "YOLO_VFOV_DEG": env_float("AIRPUFF_YOLO_VFOV_DEG", 38.0),
    "YOLO_HARD_STOP_ON_CENTER": env_bool("AIRPUFF_YOLO_HARD_STOP_ON_CENTER", True),
    "YOLO_CENTER_STOP_OVERLAP": env_float("AIRPUFF_YOLO_CENTER_STOP_OVERLAP", 0.18),
    "YOLO_OBSTACLE_CONF_MIN": env_float("AIRPUFF_YOLO_OBSTACLE_CONF_MIN", 0.42),
    "YOLO_HIGH_CONF_BYPASS": env_float("AIRPUFF_YOLO_HIGH_CONF_BYPASS", 0.68),
    "YOLO_CONFIRM_HITS": env_int("AIRPUFF_YOLO_CONFIRM_HITS", 2),
    "YOLO_MAX_AREA_RATIO": env_float("AIRPUFF_YOLO_MAX_AREA_RATIO", 0.72),
    "YOLO_EDGE_MARGIN": env_float("AIRPUFF_YOLO_EDGE_MARGIN", 0.02),
    "YOLO_CENTER_ANCHOR_MIN": env_float("AIRPUFF_YOLO_CENTER_ANCHOR_MIN", 0.34),
    "YOLO_CENTER_ANCHOR_MAX": env_float("AIRPUFF_YOLO_CENTER_ANCHOR_MAX", 0.66),
    "DEPTH_ANYTHING_ENABLED": env_bool("AIRPUFF_DEPTH_ANYTHING_ENABLED", False),
    "DEPTH_ANYTHING_MODEL": os.getenv(
        "AIRPUFF_DEPTH_ANYTHING_MODEL",
        "depth-anything/Depth-Anything-V2-Small-hf",
    ),
    "DEPTH_ANYTHING_DEVICE": os.getenv("AIRPUFF_DEPTH_ANYTHING_DEVICE", "").strip().lower(),
    "DEPTH_ANYTHING_MIN_DISTANCE_M": env_float("AIRPUFF_DEPTH_ANYTHING_MIN_DISTANCE_M", 0.3),
    "DEPTH_ANYTHING_MAX_DISTANCE_M": env_float("AIRPUFF_DEPTH_ANYTHING_MAX_DISTANCE_M", 25.0),
    "DEPTH_ANYTHING_SCALE_SMOOTHING": env_float("AIRPUFF_DEPTH_ANYTHING_SCALE_SMOOTHING", 0.35),
    "DEPTH_ANYTHING_ROI_X_MARGIN": env_float("AIRPUFF_DEPTH_ANYTHING_ROI_X_MARGIN", 0.18),
    "DEPTH_ANYTHING_ROI_Y_TOP": env_float("AIRPUFF_DEPTH_ANYTHING_ROI_Y_TOP", 0.38),
    "DEPTH_ANYTHING_ROI_Y_BOTTOM": env_float("AIRPUFF_DEPTH_ANYTHING_ROI_Y_BOTTOM", 0.95),
    "DEPTH_ANYTHING_TRIM_FRAC": env_float("AIRPUFF_DEPTH_ANYTHING_TRIM_FRAC", 0.1),
    "YOLO_OBSTACLE_LABELS": env_csv(
        "AIRPUFF_YOLO_OBSTACLE_LABELS",
        (
            "person,bicycle,motorcycle,car,bus,truck,chair,bench,dog,cat,potted_plant,"
            "backpack,suitcase,bottle,cup,remote,cell_phone,teddy_bear,book,laptop,keyboard,mouse"
        ),
    ),
    "YOLO_OCCUPANCY_STOP_AREA_RATIO": env_float("AIRPUFF_YOLO_OCCUPANCY_STOP_AREA_RATIO", 0.055),
    "YOLO_APPROACH_ENABLED": env_bool("AIRPUFF_YOLO_APPROACH_ENABLED", True),
    "YOLO_APPROACH_IOU_MIN": env_float("AIRPUFF_YOLO_APPROACH_IOU_MIN", 0.2),
    "YOLO_APPROACH_WINDOW_SEC": env_float("AIRPUFF_YOLO_APPROACH_WINDOW_SEC", 1.2),
    "YOLO_APPROACH_GROWTH_RATIO": env_float("AIRPUFF_YOLO_APPROACH_GROWTH_RATIO", 1.18),
    "YOLO_APPROACH_GROWTH_DELTA": env_float("AIRPUFF_YOLO_APPROACH_GROWTH_DELTA", 0.008),
    "YOLO_APPROACH_CENTER_OVERLAP": env_float("AIRPUFF_YOLO_APPROACH_CENTER_OVERLAP", 0.16),
    "YOLO_FLOW_FALLBACK": env_bool("AIRPUFF_YOLO_FLOW_FALLBACK", True),
    "VISION_SCAN_ENABLED": env_bool("AIRPUFF_VISION_SCAN", True),
    "VISION_DET_SYNC_MAX_LAG_SEC": env_float("AIRPUFF_VISION_DET_SYNC_MAX_LAG_SEC", 0.75),
    "VISION_SCAN_DURATION_MS": env_int("AIRPUFF_SCAN_DURATION_MS", 1200),
    "VISION_SCAN_STEP_MS": env_int("AIRPUFF_SCAN_STEP_MS", 300),
    "VISION_DIFF_TH": env_float("AIRPUFF_VISION_DIFF_TH", 0.03),
    "VISION_TURN_HOLD_MS": env_int("AIRPUFF_TURN_HOLD_MS", 600),
    "OLLAMA_NUM_PREDICT": env_int("AIRPUFF_NUM_PREDICT", 16),
    "ENABLE_KEYWORD_FALLBACK": env_bool("AIRPUFF_KEYWORD_FALLBACK", True),
    "VOICE_WAKE_ENABLED": env_bool("AIRPUFF_VOICE_WAKE_ENABLED", True),
    "VOICE_WAKE_WORDS_RAW": os.getenv(
        "AIRPUFF_VOICE_WAKE_WORDS",
        "你好飞艇,飞艇飞艇,飞艇,hello airpuff,hi airpuff",
    ),
    "AUTOPILOT_IDLE_SEC": env_int("AIRPUFF_AUTOPILOT_IDLE_SEC", 5),
    "AUTOPILOT_MODE": os.getenv("AIRPUFF_AUTOPILOT_MODE", "wander"),  # wander | circle | off
    "AUTOPILOT_STEP_SEC": env_int("AIRPUFF_AUTOPILOT_STEP_SEC", 2),
    "AUTOPILOT_FORWARD_STEPS": env_int("AIRPUFF_AUTOPILOT_FORWARD_STEPS", 5),
    "AUTOPILOT_TURN_ACTION": os.getenv("AIRPUFF_AUTOPILOT_TURN_ACTION", "RIGHT"),
    "STOP_CONDITION": os.getenv("AIRPUFF_STOP_CONDITION", "A").strip().upper() or "A",
    "CHAT_HISTORY_MAX": env_int("AIRPUFF_CHAT_HISTORY_MAX", 6),
    "CHAT_NUM_PREDICT": env_int("AIRPUFF_CHAT_NUM_PREDICT", 48),
    "LOG_PATH": os.getenv("AIRPUFF_LOG_PATH", ""),
    "LOG_IMAGES": env_bool("AIRPUFF_LOG_IMAGES", False),
    "LOG_AUDIO": env_bool("AIRPUFF_LOG_AUDIO", False),
    "LOG_TEXT": env_bool("AIRPUFF_LOG_TEXT", True),
    "CAMERA_CONTROL": env_bool("AIRPUFF_CAMERA_CONTROL", True),
    "CAMERA_SECRET_PATH": os.getenv("AIRPUFF_CAMERA_SECRET_PATH", os.path.expanduser("~/.airpuff_camera_secret.json")),
    "CAMERA_SSH_HOST": os.getenv("AIRPUFF_CAMERA_SSH_HOST", ""),
    "CAMERA_SSH_USER": os.getenv("AIRPUFF_CAMERA_SSH_USER", ""),
    "CAMERA_SSH_PASSWORD": os.getenv("AIRPUFF_CAMERA_SSH_PASSWORD", ""),
    "CAMERA_SERVICE": os.getenv("AIRPUFF_CAMERA_SERVICE", "pi_airpuff_video_stream.service"),
    "CAMERA_STATUS_CACHE_SEC": env_float("AIRPUFF_CAMERA_STATUS_CACHE_SEC", 2.5),
    "CAMERA_STREAM_TIMEOUT_SEC": env_float("AIRPUFF_CAMERA_STREAM_TIMEOUT_SEC", 2.8),
    "CAMERA_SSH_TIMEOUT_SEC": env_float("AIRPUFF_CAMERA_SSH_TIMEOUT_SEC", 15.0),
    "SERVER_HOST": os.getenv("AIRPUFF_HOST", "0.0.0.0"),
    "SERVER_PORT": env_int("AIRPUFF_PORT", 5000),
}

if CONFIG["STOP_CONDITION"] not in {"A"}:
    CONFIG["STOP_CONDITION"] = "A"

if not CONFIG["DEPTH_ANYTHING_DEVICE"]:
    CONFIG["DEPTH_ANYTHING_DEVICE"] = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

CONFIG["VOICE_WAKE_WORDS"] = [
    item.strip()
    for item in str(CONFIG["VOICE_WAKE_WORDS_RAW"]).split(",")
    if item.strip()
]
if not CONFIG["VOICE_WAKE_WORDS"]:
    CONFIG["VOICE_WAKE_WORDS"] = ["你好飞艇", "飞艇飞艇", "飞艇", "hello airpuff", "hi airpuff"]


state_lock = threading.Lock()
state: Dict[str, Any] = {
    "mode": "AUTO",  # "AUTO" or "MANUAL"
    "manual_cmd": "STOP",
    "latest_image": "",
    "latest_image_ts": 0.0,
    "latest_transcript": "",
    "latest_chat": "",
    "latest_action": "STOP",
    "altitude_setpoint": 0,
    "last_asr_ms": None,
    "last_llm_ms": None,
    "last_vlm_ms": None,
    "latest_wake_hit": False,
    "latest_wake_word": "",
    "latest_wake_required": False,
    "last_route": "",
    "last_error": "",
    "last_input_ts": time.time(),
    "chat_history": [],
    "vision_debug": {},
}

metrics_lock = threading.Lock()
metrics: Dict[str, Any] = {
    "requests_total": 0,
    "asr_calls": 0,
    "llm_calls": 0,
    "llm_cmd_calls": 0,
    "llm_chat_calls": 0,
    "vlm_calls": 0,
    "fast_cmd_hits": 0,
    "keyword_fallback_hits": 0,
    "errors": 0,
    "latency_ms": {"asr": [], "llm": [], "vlm": []},
    "last_vlm_ts": 0.0,
    "seq": 0,
}

vision_state: Dict[str, Any] = {
    "last_frame": None,
    "last_lite_ts": 0.0,
    "last_flow_ts": 0.0,
    "last_yolo_ts": 0.0,
    "last_turn": "",
    "last_turn_ts": 0.0,
    "scan_active": False,
    "scan_dir": "LEFT",
    "scan_start_ts": 0.0,
    "scan_next_switch_ts": 0.0,
    "scan_scores": {"LEFT": 999.0, "RIGHT": 999.0},
    "last_vision_mode": "",
    "last_bright_frac": None,
    "last_obstacle_th_base": None,
    "last_obstacle_th": None,
    "last_center_val": None,
    "last_left_val": None,
    "last_right_val": None,
    "last_nearest_distance_m": None,
    "last_detections": [],
    "last_detections_held": False,
    "last_depth_scale_m_per_unit": None,
    "last_depth_runtime": "",
    "last_yolo_fallback_used": False,
    "last_yolo_fallback_reason": "",
    "last_approach_hit": False,
    "last_approach_reason": "",
    "last_obstacle_tracks": [],
    "yolo_empty_streak": 0,
    "last_nonempty_detections": [],
    "last_nonempty_detection_server_ts": 0.0,
    "last_detection_image_ts": 0.0,
    "last_detection_server_ts": 0.0,
    "last_lane_markings_visible": False,
    "last_front_blocked": False,
    "last_emergency_stop": False,
    "last_emergency_distance_m": None,
    "last_stop_reason": "",
    "detector_ready": False,
    "detector_error": "",
    "depth_ready": False,
    "depth_error": "",
    "last_vision_ts": 0.0,
}

camera_state_lock = threading.Lock()
camera_state: Dict[str, Any] = {
    "control_enabled": CONFIG["CAMERA_CONTROL"],
    "configured": False,
    "host": "",
    "service": CONFIG["CAMERA_SERVICE"],
    "reachable": None,
    "service_active": None,
    "service_enabled": None,
    "active_state": "unknown",
    "unit_file_state": "unknown",
    "last_error": "",
    "last_refresh_ts": 0.0,
}


def record_latency(name: str, ms: float) -> None:
    with metrics_lock:
        metrics["latency_ms"].setdefault(name, []).append(ms)


def append_log(entry: Dict[str, Any]) -> None:
    path = CONFIG["LOG_PATH"]
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception:
        pass


LOCAL_CAMERA_HOSTS = {"", "127.0.0.1", "localhost"}
ACTIVE_CAMERA_STATES = {"active", "activating", "reloading"}
ENABLED_CAMERA_STATES = {"enabled", "static", "indirect"}


def _load_camera_secret(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _camera_settings() -> Dict[str, Any]:
    secret = _load_camera_secret(CONFIG["CAMERA_SECRET_PATH"])
    host = str(CONFIG["CAMERA_SSH_HOST"] or secret.get("host") or "").strip()
    user = str(CONFIG["CAMERA_SSH_USER"] or secret.get("user") or "").strip()
    password = str(CONFIG["CAMERA_SSH_PASSWORD"] or secret.get("password") or "").strip()
    service = str(CONFIG["CAMERA_SERVICE"] or secret.get("service") or "pi_airpuff_video_stream.service").strip()
    local = host in LOCAL_CAMERA_HOSTS
    configured = bool(CONFIG["CAMERA_CONTROL"] and service and (local or (host and user)))
    return {
        "host": host,
        "user": user,
        "password": password,
        "service": service,
        "local": local,
        "configured": configured,
    }


def _camera_stream_snapshot() -> Dict[str, Any]:
    with state_lock:
        latest_image = bool(state.get("latest_image"))
        last_image_ts = float(state.get("latest_image_ts") or 0.0)
    age_ms = round((time.time() - last_image_ts) * 1000.0) if last_image_ts else None
    streaming = bool(
        latest_image
        and last_image_ts
        and (time.time() - last_image_ts) <= max(0.6, float(CONFIG["CAMERA_STREAM_TIMEOUT_SEC"]))
    )
    return {
        "streaming": streaming,
        "last_frame_age_ms": age_ms,
    }


def _camera_status_snapshot() -> Dict[str, Any]:
    with camera_state_lock:
        snapshot = dict(camera_state)
    snapshot.update(_camera_stream_snapshot())
    last_refresh_ts = float(snapshot.get("last_refresh_ts") or 0.0)
    snapshot["last_refresh_age_ms"] = round((time.time() - last_refresh_ts) * 1000.0) if last_refresh_ts else None
    return snapshot


def _parse_camera_status_output(output: str) -> Dict[str, Any]:
    active_state = "unknown"
    unit_file_state = "unknown"
    for raw_line in output.splitlines():
        line = raw_line.strip().replace("\r", "")
        if line.startswith("ACTIVE="):
            active_state = line.split("=", 1)[1].strip() or "unknown"
        elif line.startswith("ENABLED="):
            unit_file_state = line.split("=", 1)[1].strip() or "unknown"
    return {
        "service_active": active_state in ACTIVE_CAMERA_STATES,
        "service_enabled": unit_file_state in ENABLED_CAMERA_STATES,
        "active_state": active_state,
        "unit_file_state": unit_file_state,
    }


def _run_interactive_command(argv: List[str], password: str = "", timeout_sec: float = 15.0) -> Tuple[int, str]:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(argv, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
    os.close(slave_fd)
    output = bytearray()
    yes_sent = 0
    password_sent = 0
    deadline = time.time() + max(1.0, float(timeout_sec))
    try:
        while True:
            if proc.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    output.extend(chunk)
                break

            remaining = deadline - time.time()
            if remaining <= 0:
                proc.kill()
                raise TimeoutError("camera_ssh_timeout")

            ready, _, _ = select.select([master_fd], [], [], min(0.25, remaining))
            if not ready:
                continue

            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                continue
            output.extend(chunk)

            recent = output[-768:].decode("utf-8", errors="ignore").lower()
            if ("yes/no" in recent or "continue connecting" in recent) and yes_sent < 1:
                os.write(master_fd, b"yes\n")
                yes_sent += 1
                continue
            if "password:" in recent and password and password_sent < 3:
                os.write(master_fd, (password + "\n").encode("utf-8"))
                password_sent += 1

        code = proc.wait(timeout=1.0)
        return code, output.decode("utf-8", errors="ignore")
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master_fd)


def _run_camera_shell(shell_command: str, settings: Dict[str, Any], timeout_sec: Optional[float] = None) -> str:
    timeout = timeout_sec if timeout_sec is not None else float(CONFIG["CAMERA_SSH_TIMEOUT_SEC"])
    if settings.get("local"):
        result = subprocess.run(
            ["bash", "-lc", shell_command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise RuntimeError(output.strip() or f"camera_shell_failed:{result.returncode}")
        return output

    remote_cmd = f"bash -lc {shlex.quote(shell_command)}"
    argv = [
        "ssh",
        "-tt",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
        f"{settings['user']}@{settings['host']}",
        remote_cmd,
    ]
    if settings.get("password"):
        code, output = _run_interactive_command(argv, password=str(settings["password"]), timeout_sec=timeout)
    else:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        code = result.returncode
        output = (result.stdout or "") + (result.stderr or "")
    if code != 0:
        raise RuntimeError(output.strip() or f"camera_ssh_failed:{code}")
    return output


def _camera_query_shell(service: str) -> str:
    svc = shlex.quote(service)
    return (
        f"printf 'ACTIVE='; systemctl is-active {svc} 2>/dev/null || true; "
        f"printf '\\nENABLED='; systemctl is-enabled {svc} 2>/dev/null || true; "
        "printf '\\n'"
    )


def refresh_camera_status(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    cache_window = max(0.25, float(CONFIG["CAMERA_STATUS_CACHE_SEC"]))
    with camera_state_lock:
        last_refresh_ts = float(camera_state.get("last_refresh_ts") or 0.0)
    if not force and last_refresh_ts and (now - last_refresh_ts) < cache_window:
        return _camera_status_snapshot()

    settings = _camera_settings()
    snapshot = {
        "control_enabled": CONFIG["CAMERA_CONTROL"],
        "configured": settings["configured"],
        "host": settings["host"] or "local",
        "service": settings["service"],
        "reachable": None,
        "service_active": None,
        "service_enabled": None,
        "active_state": "unknown",
        "unit_file_state": "unknown",
        "last_error": "",
        "last_refresh_ts": now,
    }
    if not CONFIG["CAMERA_CONTROL"]:
        snapshot["last_error"] = "camera_control_disabled"
    elif not settings["configured"]:
        snapshot["last_error"] = "camera_control_not_configured"
    else:
        try:
            query_timeout = min(float(CONFIG["CAMERA_SSH_TIMEOUT_SEC"]), 10.0)
            output = _run_camera_shell(_camera_query_shell(settings["service"]), settings, timeout_sec=query_timeout)
            snapshot.update(_parse_camera_status_output(output))
            snapshot["reachable"] = True
        except Exception as exc:
            snapshot["reachable"] = False
            snapshot["last_error"] = str(exc)

    with camera_state_lock:
        camera_state.update(snapshot)
    return _camera_status_snapshot()


def build_vision_debug(vision_action: str = "", vision_route: str = "") -> Dict[str, Any]:
    stream_snapshot = _camera_stream_snapshot()
    return {
        "mode": vision_state.get("last_vision_mode"),
        "bright_frac": vision_state.get("last_bright_frac"),
        "obstacle_th_base": vision_state.get("last_obstacle_th_base"),
        "obstacle_th": vision_state.get("last_obstacle_th"),
        "center_val": vision_state.get("last_center_val"),
        "left_val": vision_state.get("last_left_val"),
        "right_val": vision_state.get("last_right_val"),
        "nearest_distance_m": vision_state.get("last_nearest_distance_m"),
        "detections": vision_state.get("last_detections", []),
        "detections_held": vision_state.get("last_detections_held", False),
        "depth_scale_m_per_unit": vision_state.get("last_depth_scale_m_per_unit"),
        "depth_runtime": vision_state.get("last_depth_runtime"),
        "fallback_used": vision_state.get("last_yolo_fallback_used", False),
        "fallback_reason": vision_state.get("last_yolo_fallback_reason", ""),
        "approach_hit": vision_state.get("last_approach_hit", False),
        "approach_reason": vision_state.get("last_approach_reason", ""),
        "yolo_empty_streak": vision_state.get("yolo_empty_streak", 0),
        "detections_image_ts": vision_state.get("last_detection_image_ts"),
        "detections_server_ts": vision_state.get("last_detection_server_ts"),
        "lane_markings_visible": vision_state.get("last_lane_markings_visible"),
        "front_blocked": vision_state.get("last_front_blocked"),
        "emergency_stop": vision_state.get("last_emergency_stop"),
        "emergency_stop_distance_m": vision_state.get("last_emergency_distance_m"),
        "emergency_stop_limit_m": CONFIG["YOLO_EMERGENCY_STOP_M"],
        "stop_reason": vision_state.get("last_stop_reason"),
        "detector_ready": vision_state.get("detector_ready"),
        "detector_error": vision_state.get("detector_error"),
        "depth_ready": vision_state.get("depth_ready"),
        "depth_error": vision_state.get("depth_error"),
        "vision_ts": vision_state.get("last_vision_ts"),
        "scan_active": vision_state.get("scan_active"),
        "scan_dir": vision_state.get("scan_dir"),
        "suggested_action": vision_action,
        "suggested_route": vision_route,
        "streaming": stream_snapshot["streaming"],
        "last_frame_age_ms": stream_snapshot["last_frame_age_ms"],
    }


def _clear_vision_runtime() -> None:
    vision_state["last_frame"] = None
    vision_state["last_lite_ts"] = 0.0
    vision_state["last_flow_ts"] = 0.0
    vision_state["last_yolo_ts"] = 0.0
    vision_state["last_turn"] = ""
    vision_state["last_turn_ts"] = 0.0
    vision_state["scan_active"] = False
    vision_state["scan_dir"] = "LEFT"
    vision_state["scan_start_ts"] = 0.0
    vision_state["scan_next_switch_ts"] = 0.0
    vision_state["scan_scores"] = {"LEFT": 999.0, "RIGHT": 999.0}
    vision_state["last_bright_frac"] = None
    vision_state["last_obstacle_th_base"] = None
    vision_state["last_obstacle_th"] = None
    vision_state["last_center_val"] = None
    vision_state["last_left_val"] = None
    vision_state["last_right_val"] = None
    vision_state["last_nearest_distance_m"] = None
    vision_state["last_detections"] = []
    vision_state["last_detections_held"] = False
    vision_state["last_depth_scale_m_per_unit"] = None
    vision_state["last_depth_runtime"] = ""
    vision_state["last_yolo_fallback_used"] = False
    vision_state["last_yolo_fallback_reason"] = ""
    vision_state["last_approach_hit"] = False
    vision_state["last_approach_reason"] = ""
    vision_state["last_obstacle_tracks"] = []
    vision_state["yolo_empty_streak"] = 0
    vision_state["last_nonempty_detections"] = []
    vision_state["last_nonempty_detection_server_ts"] = 0.0
    vision_state["last_detection_image_ts"] = 0.0
    vision_state["last_detection_server_ts"] = 0.0
    vision_state["last_lane_markings_visible"] = False
    vision_state["last_front_blocked"] = False
    vision_state["last_emergency_stop"] = False
    vision_state["last_emergency_distance_m"] = None
    vision_state["last_stop_reason"] = ""
    vision_state["depth_error"] = ""
    vision_state["last_vision_ts"] = 0.0
    with state_lock:
        state["latest_image"] = ""
        state["latest_image_ts"] = 0.0
    cleared_debug = build_vision_debug()
    with state_lock:
        state["vision_debug"] = cleared_debug


def camera_service_action(action: str) -> Dict[str, Any]:
    action = str(action or "status").strip().lower()
    if action == "toggle":
        current = refresh_camera_status(force=True)
        action = "stop" if current.get("service_active") else "start"
    if action == "status":
        return {"ok": True, "action": action, "camera": refresh_camera_status(force=True)}
    if action not in {"start", "stop", "restart"}:
        return {"ok": False, "action": action, "error": "invalid_camera_action", "camera": refresh_camera_status(force=False)}

    settings = _camera_settings()
    if not settings["configured"]:
        camera = refresh_camera_status(force=True)
        return {"ok": False, "action": action, "error": "camera_control_not_configured", "camera": camera}

    _clear_vision_runtime()
    svc = shlex.quote(settings["service"])
    sudo_prefix = f"echo {shlex.quote(settings['password'])} | sudo -S" if settings.get("password") else "sudo"
    if action == "start":
        shell_command = f"{sudo_prefix} systemctl start {svc}; {_camera_query_shell(settings['service'])}"
    elif action == "restart":
        shell_command = f"{sudo_prefix} systemctl restart {svc}; {_camera_query_shell(settings['service'])}"
    else:
        shell_command = (
            f"{sudo_prefix} systemctl stop {svc} || true; "
            "pkill -f '^python3 /home/walter/airpuff/pi_video_stream_push.py$' || true; "
            "pkill -x rpicam-vid || true; "
            f"{_camera_query_shell(settings['service'])}"
        )

    try:
        output = _run_camera_shell(shell_command, settings)
        snapshot = {
            "control_enabled": CONFIG["CAMERA_CONTROL"],
            "configured": settings["configured"],
            "host": settings["host"] or "local",
            "service": settings["service"],
            "reachable": True,
            "last_error": "",
            "last_refresh_ts": time.time(),
        }
        snapshot.update(_parse_camera_status_output(output))
        with camera_state_lock:
            camera_state.update(snapshot)
        return {"ok": True, "action": action, "camera": _camera_status_snapshot()}
    except Exception as exc:
        with camera_state_lock:
            camera_state.update(
                {
                    "control_enabled": CONFIG["CAMERA_CONTROL"],
                    "configured": settings["configured"],
                    "host": settings["host"] or "local",
                    "service": settings["service"],
                    "reachable": False,
                    "last_error": str(exc),
                    "last_refresh_ts": time.time(),
                }
            )
        return {"ok": False, "action": action, "error": str(exc), "camera": _camera_status_snapshot()}


YOLO_SIZE_PRIORS = {
    "person": ("h", 1.70),
    "bicycle": ("h", 1.10),
    "motorcycle": ("h", 1.20),
    "car": ("w", 1.80),
    "bus": ("w", 2.55),
    "truck": ("w", 2.60),
    "bench": ("w", 1.40),
    "chair": ("h", 0.95),
    "dog": ("h", 0.55),
    "cat": ("h", 0.28),
    "potted_plant": ("h", 0.65),
    "backpack": ("h", 0.50),
    "suitcase": ("h", 0.70),
    "bottle": ("h", 0.28),
    "cup": ("h", 0.10),
    "remote": ("w", 0.05),
    "cell_phone": ("h", 0.15),
    "teddy_bear": ("h", 0.26),
    "book": ("h", 0.24),
    "laptop": ("w", 0.33),
    "keyboard": ("w", 0.44),
    "mouse": ("w", 0.06),
}

YOLO_RENDER_KIND = {
    "person": "person",
    "dog": "person",
    "cat": "person",
    "potted_plant": "tree",
    "chair": "seat",
    "bench": "seat",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
}

SR_ASSET_LIBRARY = {
    "ground_road": {
        "family": "ground",
        "model": "",
        "material": "PhysicalModel/Ground/Material/MI_RoadColor_01",
    },
    "ground_lane_primary": {
        "family": "ground",
        "model": "",
        "material": "PhysicalModel/Ground/Material/MI_LaneLine_01",
    },
    "ground_lane_secondary": {
        "family": "ground",
        "model": "",
        "material": "PhysicalModel/Ground/Material/MI_LaneLine_02",
    },
    "ground_grass": {
        "family": "ground",
        "model": "",
        "material": "PhysicalModel/Ground/Material/MI_Grass_01",
    },
    "ego_vehicle": {
        "family": "vehicle",
        "model": "PhysicalModel/MainCar/Model/SM_MainCar_01",
        "material": "PhysicalModel/MainCar/Material/MI_Paint_01",
    },
    "det_vehicle": {
        "family": "vehicle",
        "model": "PhysicalModel/MainCar/Model/SM_MainCar_01",
        "material": "PhysicalModel/MainCar/Material/MI_Paint_01",
    },
    "guide_forward": {
        "family": "guide_arrow",
        "model": "NavigationElements/GuideArrow/SM_GuideArrow_L2",
        "material": "NavigationElements/GuideArrow/MI_GuideArrow_L2",
    },
    "guide_left": {
        "family": "guide_arrow",
        "model": "NavigationElements/GuideArrow/SM_GuideArrow_L1",
        "material": "NavigationElements/GuideArrow/MI_GuideArrow_L1",
    },
    "guide_right": {
        "family": "guide_arrow",
        "model": "NavigationElements/GuideArrow/SM_GuideArrow_L3",
        "material": "NavigationElements/GuideArrow/MI_GuideArrow_L3",
    },
    "guide_change_lane": {
        "family": "guide_arrow",
        "model": "NavigationElements/GuideArrow/SM_GuideArrow_ChangeLane",
        "material": "NavigationElements/GuideArrow/MI_GuideArrow_L2_ChangeLane",
    },
    "guide_idle": {
        "family": "guide_arrow",
        "model": "NavigationElements/GuideArrow/SM_GuideArrow_NoCom",
        "material": "NavigationElements/GuideArrow/MI_GuideArrow_01_Blue_NoCom",
    },
    "guardrail": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_SideGuardrail_01",
        "material": "PhysicalModel/Common/Material/MI_BaseColor_01",
    },
    "pier_primary": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_Pier_01",
        "material": "PhysicalModel/Common/Material/MI_BaseColor_02",
    },
    "pier_secondary": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_Pier_02",
        "material": "PhysicalModel/Common/Material/MI_BaseColor_03",
    },
    "gantry_primary": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_LongmenFrame_01",
        "material": "PhysicalModel/Common/Material/MI_BaseColor_01",
    },
    "gantry_secondary": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_LongmenFrame_02",
        "material": "PhysicalModel/Common/Material/MI_BaseColor_02",
    },
    "speed_limit": {
        "family": "roadside",
        "model": "PhysicalModel/Common/Model/SM_SpeedLimitBoard_01",
        "material": "PhysicalModel/Common/Material/MI_SpeedLimitBoard_01",
    },
    "building_low": {
        "family": "building",
        "model": "PhysicalModel/Building/IntermediateLevelBuilding/Model/M0-1_1_0-1_4_20-40_1_3000_2500_692",
        "material": "PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_01",
    },
    "building_mid": {
        "family": "building",
        "model": "PhysicalModel/Building/IntermediateLevelBuilding/Model/M2-5_2_6-3_0_20-40_1_8400_3000_2087",
        "material": "PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_03",
    },
    "building_high": {
        "family": "building",
        "model": "PhysicalModel/Building/IntermediateLevelBuilding/Model/M16-20_3_1_41_1_14400_4500_8153",
        "material": "PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_06",
    },
}

YOLO_ASSET_HINTS = {
    "car": SR_ASSET_LIBRARY["det_vehicle"],
    "bus": SR_ASSET_LIBRARY["det_vehicle"],
    "truck": SR_ASSET_LIBRARY["det_vehicle"],
    "motorcycle": SR_ASSET_LIBRARY["det_vehicle"],
    "bicycle": SR_ASSET_LIBRARY["det_vehicle"],
    "potted_plant": SR_ASSET_LIBRARY["ground_grass"],
}

YOLO_LANE_LABEL_PREFIXES = (
    "lane_",
    "road_marking",
    "road_line",
    "lane_line",
    "lane_marking",
    "stop_line",
    "center_line",
    "edge_line",
)
YOLO_LANE_LABEL_PARTS = {"lane", "marking", "crosswalk"}


def _normalize_label(label: Any) -> str:
    return str(label or "").strip().lower().replace(" ", "_")


def _is_lane_like_label(label: str) -> bool:
    name = _normalize_label(label)
    if not name:
        return False
    if name.startswith(YOLO_LANE_LABEL_PREFIXES):
        return True
    return bool(YOLO_LANE_LABEL_PARTS.intersection(name.split("_")))


def _load_bgr_image(image_b64: str) -> Optional["np.ndarray"]:
    if cv2 is None or np is None or not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _focal_px(span_px: float, fov_deg: float) -> float:
    safe_fov = max(1.0, min(float(fov_deg), 179.0))
    return float(span_px) / (2.0 * math.tan(math.radians(safe_fov) / 2.0))


def _estimate_yolo_distance(label: str, box_w: float, box_h: float, img_w: int, img_h: int) -> Optional[float]:
    spec = YOLO_SIZE_PRIORS.get(label)
    if not spec:
        return None
    axis, real_size_m = spec
    if axis == "w":
        px_size = max(1.0, box_w)
        focal = _focal_px(img_w, CONFIG["YOLO_HFOV_DEG"])
    else:
        px_size = max(1.0, box_h)
        focal = _focal_px(img_h, CONFIG["YOLO_VFOV_DEG"])
    dist = (float(real_size_m) * focal) / px_size
    return max(0.3, min(float(dist), 25.0))


def _lane_overlap(x1: float, x2: float, lane_start: float, lane_end: float) -> float:
    overlap = max(0.0, min(x2, lane_end) - max(x1, lane_start))
    width = max(1.0, x2 - x1)
    return overlap / width


def _prepare_yolo_detection(
    label: str,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    dist_m = _estimate_yolo_distance(label, box_w, box_h, img_w, img_h)
    render_kind = YOLO_RENDER_KIND.get(label, "seat")
    asset_hint = YOLO_ASSET_HINTS.get(label, {})
    lane_spans = {
        "LEFT": _lane_overlap(x1, x2, 0.0, img_w / 3.0),
        "CENTER": _lane_overlap(x1, x2, img_w / 3.0, (img_w * 2.0) / 3.0),
        "RIGHT": _lane_overlap(x1, x2, (img_w * 2.0) / 3.0, float(img_w)),
    }
    lane = max(lane_spans, key=lane_spans.get)
    depth_term = 1.0 / max(dist_m or 8.0, 0.4)
    area_ratio = (box_w * box_h) / max(float(img_w * img_h), 1.0)
    threat = float(conf) * depth_term * (0.75 + area_ratio * 3.0)
    return {
        "label": label,
        "conf": round(float(conf), 3),
        "box": [
            round(x1 / img_w, 4),
            round(y1 / img_h, 4),
            round(x2 / img_w, 4),
            round(y2 / img_h, 4),
        ],
        "box_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "box_size_px": {"w": round(box_w, 1), "h": round(box_h, 1)},
        "area_ratio": round((box_w * box_h) / max(float(img_w * img_h), 1.0), 5),
        "center_x": round(((x1 + x2) * 0.5) / max(float(img_w), 1.0), 4),
        "bottom_y": round(y2 / max(float(img_h), 1.0), 4),
        "image_size": [int(img_w), int(img_h)],
        "distance_m": round(float(dist_m), 2) if dist_m is not None else None,
        "distance_geom_m": round(float(dist_m), 2) if dist_m is not None else None,
        "approx_distance": dist_m is not None,
        "depth_rel": None,
        "distance_source": "yolo_size_prior" if dist_m is not None else "",
        "stable_hits": 1,
        "actionable": False,
        "lane": lane,
        "lane_scores": {key: round(val, 3) for key, val in lane_spans.items()},
        "render_kind": render_kind,
        "asset_family": asset_hint.get("family", ""),
        "asset_model": asset_hint.get("model", ""),
        "asset_material": asset_hint.get("material", ""),
        "threat": round(threat, 4),
    }


def _update_detection_threat(det: Dict[str, Any]) -> None:
    box_size = det.get("box_size_px") or {}
    image_size = det.get("image_size") or [1, 1]
    try:
        box_w = max(1.0, float(box_size.get("w", 1.0)))
        box_h = max(1.0, float(box_size.get("h", 1.0)))
        img_w = max(1.0, float(image_size[0]))
        img_h = max(1.0, float(image_size[1]))
    except Exception:
        return
    dist_m = det.get("distance_m")
    depth_term = 1.0 / max(float(dist_m), 0.4) if isinstance(dist_m, (int, float)) else 1.0 / 8.0
    area_ratio = (box_w * box_h) / max(img_w * img_h, 1.0)
    det["threat"] = round(float(det.get("conf", 0.0)) * depth_term * (0.75 + area_ratio * 3.0), 4)


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


def _match_previous_obstacle_track(det: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
    tracks = vision_state.get("last_obstacle_tracks") or []
    best_track = None
    best_iou = 0.0
    window_sec = max(0.2, float(CONFIG["YOLO_APPROACH_WINDOW_SEC"]))
    label = str(det.get("label") or "")
    for track in tracks:
        if str(track.get("label") or "") != label:
            continue
        ts = float(track.get("ts") or 0.0)
        if now - ts > window_sec:
            continue
        iou = _box_iou(det.get("box_px") or [], track.get("box_px") or [])
        if iou >= max(float(CONFIG["YOLO_APPROACH_IOU_MIN"]), best_iou):
            best_track = track
            best_iou = iou
    return best_track


def _is_plausible_obstacle_detection(det: Dict[str, Any]) -> bool:
    conf = float(det.get("conf") or 0.0)
    if conf < float(CONFIG["YOLO_OBSTACLE_CONF_MIN"]):
        return False
    area_ratio = float(det.get("area_ratio") or 0.0)
    if area_ratio > float(CONFIG["YOLO_MAX_AREA_RATIO"]):
        return False
    edge_margin = float(CONFIG["YOLO_EDGE_MARGIN"])
    x1, y1, x2, y2 = [float(v) for v in (det.get("box") or [0.0, 0.0, 1.0, 1.0])]
    touches_edge = (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= (1.0 - edge_margin)
        or y2 >= (1.0 - edge_margin)
    )
    # Reject giant border-hugging boxes, which are often full-frame hallucinations.
    if touches_edge and area_ratio >= 0.45 and conf < max(float(CONFIG["YOLO_HIGH_CONF_BYPASS"]), 0.75):
        return False
    return True


def _is_actionable_obstacle(det: Dict[str, Any]) -> bool:
    if not det.get("is_obstacle"):
        return False
    if not _is_plausible_obstacle_detection(det):
        return False
    conf = float(det.get("conf") or 0.0)
    hits = int(det.get("stable_hits") or 0)
    return conf >= float(CONFIG["YOLO_HIGH_CONF_BYPASS"]) or hits >= int(CONFIG["YOLO_CONFIRM_HITS"])


def _is_center_stop_candidate(det: Dict[str, Any]) -> bool:
    if not _is_actionable_obstacle(det):
        return False
    center_x = float(det.get("center_x") or 0.0)
    if not (float(CONFIG["YOLO_CENTER_ANCHOR_MIN"]) <= center_x <= float(CONFIG["YOLO_CENTER_ANCHOR_MAX"])):
        return False
    center_overlap = float((det.get("lane_scores") or {}).get("CENTER", 0.0) or 0.0)
    return center_overlap >= max(0.08, float(CONFIG["YOLO_CENTER_STOP_OVERLAP"]))


def _annotate_yolo_stability(detections: List[Dict[str, Any]], now: float) -> None:
    for det in detections:
        prev_track = _match_previous_obstacle_track(det, now)
        hits = 1
        if prev_track:
            hits = int(prev_track.get("hits") or 0) + 1
        det["stable_hits"] = hits
        det["actionable"] = _is_actionable_obstacle(det)


def _annotate_yolo_approach(detections: List[Dict[str, Any]], now: float) -> Optional[Dict[str, Any]]:
    if not CONFIG["YOLO_APPROACH_ENABLED"]:
        return None
    approach_hit = None
    for det in detections:
        det["approaching"] = False
        det["approach_ratio"] = None
        det["approach_dt_ms"] = None
        if not _is_actionable_obstacle(det):
            continue
        center_overlap = float((det.get("lane_scores") or {}).get("CENTER", 0.0) or 0.0)
        if center_overlap < float(CONFIG["YOLO_APPROACH_CENTER_OVERLAP"]):
            continue
        prev_track = _match_previous_obstacle_track(det, now)
        if not prev_track:
            continue
        prev_area = float(prev_track.get("area_ratio") or 0.0)
        curr_area = float(det.get("area_ratio") or 0.0)
        dt_ms = max(1.0, (now - float(prev_track.get("ts") or now)) * 1000.0)
        if prev_area <= 0.0 or curr_area <= prev_area:
            continue
        growth_ratio = curr_area / max(prev_area, 1e-6)
        growth_delta = curr_area - prev_area
        det["approach_ratio"] = round(growth_ratio, 3)
        det["approach_dt_ms"] = int(round(dt_ms))
        if (
            growth_ratio >= float(CONFIG["YOLO_APPROACH_GROWTH_RATIO"])
            and growth_delta >= float(CONFIG["YOLO_APPROACH_GROWTH_DELTA"])
        ):
            det["approaching"] = True
            if approach_hit is None or curr_area > float(approach_hit.get("area_ratio") or 0.0):
                approach_hit = det
    return approach_hit


def _remember_obstacle_tracks(detections: List[Dict[str, Any]], now: float) -> None:
    tracks: List[Dict[str, Any]] = []
    for det in detections:
        if not det.get("is_obstacle"):
            continue
        tracks.append(
            {
                "label": det.get("label"),
                "box_px": list(det.get("box_px") or []),
                "area_ratio": float(det.get("area_ratio") or 0.0),
                "hits": int(det.get("stable_hits") or 1),
                "ts": float(now),
            }
        )
    vision_state["last_obstacle_tracks"] = tracks


def _depth_anything_available() -> bool:
    return bool(
        CONFIG["DEPTH_ANYTHING_ENABLED"]
        and torch is not None
        and Image is not None
        and AutoImageProcessor is not None
        and AutoModelForDepthEstimation is not None
        and depth_model is not None
        and depth_processor is not None
    )


def _predict_depth_map(img: "np.ndarray") -> Tuple[Optional["np.ndarray"], str]:
    if cv2 is None or np is None:
        return None, "opencv_missing"
    if not _depth_anything_available():
        return None, depth_error or "depth_anything_unavailable"
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = depth_processor(images=pil_img, return_tensors="pt")
        device = str(CONFIG["DEPTH_ANYTHING_DEVICE"] or "cpu")
        if device != "cpu":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = depth_model(**inputs)
        processed = depth_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(img.shape[0], img.shape[1])],
        )
        depth = processed[0]["predicted_depth"]
        if hasattr(depth, "detach"):
            depth = depth.detach().float().cpu().numpy()
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim != 2:
            return None, "depth_map_invalid"
        return depth_arr, ""
    except Exception as exc:
        return None, str(exc)


def _trimmed_median(values: "np.ndarray", trim_frac: float) -> Optional[float]:
    if np is None:
        return None
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    flat = flat[flat > 0]
    if flat.size == 0:
        return None
    flat.sort()
    trim = int(flat.size * _clamp(float(trim_frac), 0.0, 0.45))
    if trim > 0 and (trim * 2) < flat.size:
        flat = flat[trim : flat.size - trim]
    return float(np.median(flat)) if flat.size else None


def _depth_roi_value(depth_map: "np.ndarray", box_px: List[float]) -> Optional[float]:
    if np is None or len(box_px) != 4:
        return None
    img_h, img_w = depth_map.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_px]
    box_w = max(2.0, x2 - x1)
    box_h = max(2.0, y2 - y1)
    x_margin = box_w * _clamp(float(CONFIG["DEPTH_ANYTHING_ROI_X_MARGIN"]), 0.0, 0.45)
    y0 = y1 + box_h * _clamp(float(CONFIG["DEPTH_ANYTHING_ROI_Y_TOP"]), 0.0, 0.95)
    y1_roi = y1 + box_h * _clamp(float(CONFIG["DEPTH_ANYTHING_ROI_Y_BOTTOM"]), 0.05, 1.0)
    rx1 = int(_clamp(math.floor(x1 + x_margin), 0, max(0, img_w - 1)))
    rx2 = int(_clamp(math.ceil(x2 - x_margin), rx1 + 1, img_w))
    ry1 = int(_clamp(math.floor(y0), 0, max(0, img_h - 1)))
    ry2 = int(_clamp(math.ceil(y1_roi), ry1 + 1, img_h))
    roi = depth_map[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    return _trimmed_median(roi, float(CONFIG["DEPTH_ANYTHING_TRIM_FRAC"]))


def _apply_depth_anything_distances(detections: List[Dict[str, Any]], depth_map: "np.ndarray") -> Optional[float]:
    scale_candidates: List[float] = []
    for det in detections:
        rel_depth = _depth_roi_value(depth_map, det.get("box_px") or [])
        det["depth_rel"] = round(float(rel_depth), 4) if rel_depth is not None else None
        geom_dist = det.get("distance_geom_m")
        if rel_depth is not None and isinstance(geom_dist, (int, float)) and rel_depth > 0:
            scale_candidates.append(float(geom_dist) / float(rel_depth))

    scale = None
    if scale_candidates:
        scale_candidates.sort()
        mid = len(scale_candidates) // 2
        if len(scale_candidates) % 2:
            scale = scale_candidates[mid]
        else:
            scale = (scale_candidates[mid - 1] + scale_candidates[mid]) / 2.0
        prev_scale = vision_state.get("last_depth_scale_m_per_unit")
        if isinstance(prev_scale, (int, float)):
            alpha = _clamp(float(CONFIG["DEPTH_ANYTHING_SCALE_SMOOTHING"]), 0.0, 1.0)
            scale = (float(prev_scale) * (1.0 - alpha)) + (float(scale) * alpha)
    else:
        prev_scale = vision_state.get("last_depth_scale_m_per_unit")
        if isinstance(prev_scale, (int, float)):
            scale = float(prev_scale)

    for det in detections:
        rel_depth = det.get("depth_rel")
        if isinstance(rel_depth, (int, float)) and isinstance(scale, (int, float)):
            dist_m = float(rel_depth) * float(scale)
            dist_m = _clamp(
                dist_m,
                float(CONFIG["DEPTH_ANYTHING_MIN_DISTANCE_M"]),
                float(CONFIG["DEPTH_ANYTHING_MAX_DISTANCE_M"]),
            )
            det["distance_m"] = round(float(dist_m), 2)
            det["approx_distance"] = True
            det["distance_source"] = "depth_anything_v2_scaled"
        elif isinstance(det.get("distance_geom_m"), (int, float)):
            det["distance_m"] = round(float(det["distance_geom_m"]), 2)
            det["approx_distance"] = True
            det["distance_source"] = "yolo_size_prior"
        elif isinstance(rel_depth, (int, float)):
            det["distance_source"] = "depth_anything_v2_relative"
        _update_detection_threat(det)

    return float(scale) if isinstance(scale, (int, float)) else None


def _yolo_front_blocking_detection(
    detections: List[Dict[str, Any]],
    max_distance_m: Optional[float] = None,
    require_distance: bool = False,
) -> Optional[Dict[str, Any]]:
    center_candidates: List[Dict[str, Any]] = []

    for det in detections:
        if not _is_center_stop_candidate(det):
            continue
        center_candidates.append(det)

    if not center_candidates:
        return None

    if max_distance_m is not None:
        within_range = [
            det
            for det in center_candidates
            if det.get("distance_m") is not None and float(det["distance_m"]) <= max(0.4, float(max_distance_m))
        ]
        if within_range:
            within_range.sort(key=lambda det: float(det["distance_m"]))
            return within_range[0]
        if require_distance:
            return None

    center_candidates.sort(
        key=lambda det: (
            -float((det.get("lane_scores") or {}).get("CENTER", 0.0) or 0.0),
            -float(det.get("threat", 0.0) or 0.0),
            -float(det.get("conf", 0.0) or 0.0),
        )
    )
    return center_candidates[0]


def safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def resolve_llm_model(purpose: str) -> str:
    if purpose == "command" and CONFIG["LLM_CMD_MODEL"]:
        return CONFIG["LLM_CMD_MODEL"]
    if purpose == "chat" and CONFIG["LLM_CHAT_MODEL"]:
        return CONFIG["LLM_CHAT_MODEL"]
    return CONFIG["LLM_MODEL"]


def ask_llm(prompt: str, purpose: str) -> Tuple[str, float]:
    start = time.perf_counter()
    model = resolve_llm_model(purpose)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": CONFIG["OLLAMA_NUM_PREDICT"]},
    }
    res = requests.post(CONFIG["OLLAMA_API"], json=payload, timeout=90).json()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return res.get("response", "").strip(), latency_ms


def ask_chat(prompt: str) -> Tuple[str, float]:
    start = time.perf_counter()
    model = resolve_llm_model("chat")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": CONFIG["CHAT_NUM_PREDICT"]},
    }
    res = requests.post(CONFIG["OLLAMA_API"], json=payload, timeout=120).json()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return res.get("response", "").strip(), latency_ms


def ask_vlm(image_b64: str, prompt: str) -> Tuple[str, float]:
    start = time.perf_counter()
    payload = {
        "model": CONFIG["VLM_MODEL"],
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"num_predict": CONFIG["OLLAMA_NUM_PREDICT"]},
    }
    res = requests.post(CONFIG["OLLAMA_API"], json=payload, timeout=120).json()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return res.get("response", "").strip(), latency_ms


def keyword_command(transcript: str) -> str:
    if not transcript:
        return ""
    text = transcript.lower()
    rules = [
        (["forward", "ahead", "go forward", "前进", "往前", "向前"], "FORWARD"),
        (["back", "backward", "reverse", "后退", "向后"], "BACKWARD"),
        (["left", "turn left", "左转", "向左"], "LEFT"),
        (["right", "turn right", "右转", "向右"], "RIGHT"),
        (["up", "ascend", "rise", "上升", "升高"], "UP"),
        (["down", "descend", "lower", "下降", "降低"], "DOWN"),
        (["stop", "halt", "停", "停止"], "STOP"),
    ]
    for keywords, cmd in rules:
        if any(k in text for k in keywords):
            return cmd
    return ""


def _wake_match_pattern(wake_word: str) -> str:
    token = str(wake_word or "").strip().lower()
    if not token:
        return ""
    # Allow flexible separators between words, e.g. "hello, airpuff".
    return re.escape(token).replace(r"\ ", r"[\s,，。.!！？:：;；\-_~]*")


def apply_voice_wake_gate(transcript: str, from_audio: bool) -> Tuple[str, Dict[str, Any]]:
    text = str(transcript or "").strip()
    info: Dict[str, Any] = {
        "required": bool(CONFIG["VOICE_WAKE_ENABLED"] and from_audio),
        "hit": False,
        "word": "",
        "prompt_only": False,
    }
    if not text:
        return "", info
    if not info["required"]:
        return text, info

    text_lower = text.lower()
    best = None
    for wake_word in CONFIG["VOICE_WAKE_WORDS"]:
        pattern = _wake_match_pattern(wake_word)
        if not pattern:
            continue
        match = re.search(pattern, text_lower)
        if not match:
            continue
        candidate = (match.start(), -(match.end() - match.start()), wake_word, match.end())
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return "", info

    _, _, hit_word, end_pos = best
    info["hit"] = True
    info["word"] = hit_word
    suffix = re.sub(r"^[\s,，。.!！？:：;；\-_~]+", "", text[end_pos:]).strip()
    if not suffix:
        info["prompt_only"] = True
        return "", info
    return suffix, info


def fast_command_candidate(transcript: str) -> str:
    if not CONFIG["CMD_FAST_PATH"]:
        return ""
    text = (transcript or "").strip()
    if not text:
        return ""
    if len(text) > max(1, CONFIG["CMD_FAST_PATH_MAX_CHARS"]):
        return ""
    if any(marker in text for marker in ["然后", "接着", "之后", "再然后"]):
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in [" and then ", " after that ", " before that ", " then "]):
        return ""
    return keyword_command(text)


def direct_chat_candidate(transcript: str) -> bool:
    text = (transcript or "").strip()
    if not text:
        return False
    return keyword_command(text) == ""


def _chat_history_text() -> str:
    history = []
    with state_lock:
        history = list(state.get("chat_history", []))
    history = history[-CONFIG["CHAT_HISTORY_MAX"] :]
    lines = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_cmd_prompt(transcript: str) -> str:
    return (
        "You are the AI brain of AirPuff, a flying airship.\n"
        "Classify the user's utterance as a flight command or a casual chat.\n"
        "Output STRICTLY valid JSON ONLY.\n"
        'Format for command: {"type": "command", "action": "FORWARD"} '
        "(Allowed actions: FORWARD, BACKWARD, LEFT, RIGHT, STOP, UP, DOWN)\n"
        'Format for chat: {"type": "chat", "reply": "<short acknowledgment>"}\n'
        f"User says: {transcript}"
    )


def build_chat_prompt(transcript: str) -> str:
    history_text = _chat_history_text()
    return (
        "You are the AI brain of AirPuff, a flying airship.\n"
        "Respond concisely and naturally. Keep responses short and friendly.\n"
        f"Conversation so far:\n{history_text}\n"
        f"User says: {transcript}\n"
        "Assistant:"
    )


def remember_chat(user_text: str, reply_text: str) -> None:
    with state_lock:
        history = state.get("chat_history", [])
        history.append({"role": "user", "content": user_text})
        if reply_text:
            history.append({"role": "assistant", "content": reply_text})
        state["chat_history"] = history[-CONFIG["CHAT_HISTORY_MAX"] :]


def autopilot_action(now_ts: float) -> str:
    if CONFIG["AUTOPILOT_MODE"] == "off":
        return "STOP"
    step = max(1, CONFIG["AUTOPILOT_STEP_SEC"])
    if CONFIG["AUTOPILOT_MODE"] == "circle":
        forward_steps = max(1, CONFIG["AUTOPILOT_FORWARD_STEPS"])
        turn_action = CONFIG["AUTOPILOT_TURN_ACTION"].upper()
        cycle_len = forward_steps + 1
        idx = int(now_ts // step) % cycle_len
        return "FORWARD" if idx < forward_steps else turn_action
    # wander (default)
    pattern = ["FORWARD", "LEFT", "FORWARD", "RIGHT", "BACKWARD", "STOP"]
    idx = int(now_ts // step) % len(pattern)
    return pattern[idx]


def _bright_fraction(gray: "np.ndarray") -> float:
    th = max(0, int(CONFIG["BRIGHT_TH"]))
    if th <= 0:
        return 0.0
    bright = (gray >= th)
    return float(bright.mean())


def _apply_lighting_compensation(gray: "np.ndarray") -> "np.ndarray":
    out = gray
    if CONFIG["VISION_CLAHE"]:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        except Exception:
            out = gray
    gamma = float(CONFIG["VISION_GAMMA"])
    if gamma and abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, table)
    return out


def lite_vision_action(image_b64: str) -> str:
    if cv2 is None or np is None or not image_b64:
        return ""
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        img = cv2.resize(img, (CONFIG["LITE_W"], CONFIG["LITE_H"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_frac = _bright_fraction(gray)
        gray = _apply_lighting_compensation(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        left = edges[:, : w // 3]
        center = edges[:, w // 3 : 2 * w // 3]
        right = edges[:, 2 * w // 3 :]
        left_density = left.mean() / 255.0
        center_density = center.mean() / 255.0
        right_density = right.mean() / 255.0
        obstacle_th_base = CONFIG["LITE_EDGE_TH"] / 100.0
        obstacle_th = obstacle_th_base
        if bright_frac >= CONFIG["BRIGHT_FRAC_TH"]:
            obstacle_th *= max(1.0, CONFIG["BRIGHT_SCALE"])
        vision_state["last_vision_mode"] = "lite"
        vision_state["last_bright_frac"] = round(float(bright_frac), 4)
        vision_state["last_obstacle_th_base"] = round(float(obstacle_th_base), 4)
        vision_state["last_obstacle_th"] = round(float(obstacle_th), 4)
        vision_state["last_center_val"] = round(float(center_density), 4)
        vision_state["last_left_val"] = round(float(left_density), 4)
        vision_state["last_right_val"] = round(float(right_density), 4)
        vision_state["last_vision_ts"] = time.time()
        return _decide_turn(center_density, left_density, right_density, obstacle_th)
    except Exception:
        return ""


def flow_vision_action(image_b64: str) -> str:
    if cv2 is None or np is None or not image_b64:
        return ""
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        img = cv2.resize(img, (CONFIG["FLOW_W"], CONFIG["FLOW_H"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_frac = _bright_fraction(gray)
        gray = _apply_lighting_compensation(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prev = vision_state.get("last_frame")
        vision_state["last_frame"] = gray
        if prev is None:
            return ""
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h, w = mag.shape
        left = mag[:, : w // 3]
        center = mag[:, w // 3 : 2 * w // 3]
        right = mag[:, 2 * w // 3 :]
        left_m = float(left.mean())
        center_m = float(center.mean())
        right_m = float(right.mean())
        obstacle_th_base = CONFIG["FLOW_MAG_TH"]
        obstacle_th = obstacle_th_base
        if bright_frac >= CONFIG["BRIGHT_FRAC_TH"]:
            obstacle_th *= max(1.0, CONFIG["BRIGHT_SCALE"])
        vision_state["last_vision_mode"] = "flow"
        vision_state["last_bright_frac"] = round(float(bright_frac), 4)
        vision_state["last_obstacle_th_base"] = round(float(obstacle_th_base), 4)
        vision_state["last_obstacle_th"] = round(float(obstacle_th), 4)
        vision_state["last_center_val"] = round(float(center_m), 4)
        vision_state["last_left_val"] = round(float(left_m), 4)
        vision_state["last_right_val"] = round(float(right_m), 4)
        vision_state["last_vision_ts"] = time.time()
        return _decide_turn(center_m, left_m, right_m, obstacle_th)
    except Exception:
        return ""


def yolo_vision_action(image_b64: str, frame_ts: Optional[float] = None) -> str:
    if cv2 is None or np is None or not image_b64:
        vision_state["detector_ready"] = False
        vision_state["detector_error"] = "opencv_missing"
        return ""
    if YOLO is None or yolo_model is None:
        vision_state["detector_ready"] = False
        vision_state["detector_error"] = yolo_error or "ultralytics_missing"
        return ""
    try:
        vision_state["last_yolo_fallback_used"] = False
        vision_state["last_yolo_fallback_reason"] = ""
        vision_state["last_approach_hit"] = False
        vision_state["last_approach_reason"] = ""
        img = _load_bgr_image(image_b64)
        if img is None:
            return ""
        img_h, img_w = img.shape[:2]
        result = yolo_model.predict(
            source=img,
            imgsz=CONFIG["YOLO_IMGSZ"],
            conf=CONFIG["YOLO_CONF"],
            max_det=CONFIG["YOLO_MAX_DETS"],
            verbose=False,
        )[0]
        names = getattr(result, "names", {}) or getattr(yolo_model, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        detections: List[Dict[str, Any]] = []
        obstacle_labels = set(CONFIG["YOLO_OBSTACLE_LABELS"])

        if boxes is not None:
            xyxy_rows = boxes.xyxy.tolist()
            conf_rows = boxes.conf.tolist()
            cls_rows = boxes.cls.tolist()
            for coords, conf, cls_idx in zip(xyxy_rows, conf_rows, cls_rows):
                x1, y1, x2, y2 = [float(v) for v in coords]
                label = _normalize_label(names.get(int(cls_idx), str(int(cls_idx))))
                det = _prepare_yolo_detection(label, float(conf), x1, y1, x2, y2, img_w, img_h)
                det["is_lane_marker"] = _is_lane_like_label(label)
                det["is_obstacle"] = label in obstacle_labels
                detections.append(det)

        depth_runtime = ""
        depth_scale = None
        if detections:
            depth_map, depth_runtime = _predict_depth_map(img)
            if depth_map is not None:
                depth_scale = _apply_depth_anything_distances(detections, depth_map)
                vision_state["depth_ready"] = True
                vision_state["depth_error"] = ""
            else:
                vision_state["depth_error"] = depth_runtime
        vision_state["last_depth_runtime"] = "depth_anything_v2" if depth_scale is not None else ""
        vision_state["last_depth_scale_m_per_unit"] = round(float(depth_scale), 4) if depth_scale is not None else None

        detections.sort(
            key=lambda item: (
                0 if item.get("distance_m") is not None else 1,
                item.get("distance_m") if item.get("distance_m") is not None else 999.0,
                -float(item.get("conf", 0.0)),
            )
        )
        detections = detections[: CONFIG["YOLO_MAX_DETS"]]
        detect_now = time.time()
        _annotate_yolo_stability(detections, detect_now)
        approach_det = _annotate_yolo_approach(detections, detect_now)
        detections_held = False
        raw_detections = detections
        if raw_detections:
            vision_state["yolo_empty_streak"] = 0
            vision_state["last_nonempty_detections"] = [dict(det) for det in raw_detections]
            vision_state["last_nonempty_detection_server_ts"] = detect_now
        else:
            empty_streak = int(vision_state.get("yolo_empty_streak") or 0) + 1
            vision_state["yolo_empty_streak"] = empty_streak
            hold_sec = max(0.0, float(CONFIG["YOLO_BOX_HOLD_SEC"]))
            miss_tol = max(0, int(CONFIG["YOLO_MISS_TOLERANCE"]))
            last_nonempty = vision_state.get("last_nonempty_detections") or []
            last_nonempty_ts = float(vision_state.get("last_nonempty_detection_server_ts") or 0.0)
            if last_nonempty and empty_streak <= miss_tol and (detect_now - last_nonempty_ts) <= hold_sec:
                detections = [dict(det) for det in last_nonempty]
                for det in detections:
                    det["held"] = True
                detections_held = True

        # Recompute aggregate values from the final detection list (raw or held).
        lane_scores = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        lane_markings_visible = False
        nearest_distance = None
        center_occupancy_det = None
        for det in detections:
            if det.get("is_lane_marker"):
                lane_markings_visible = True
            if not _is_actionable_obstacle(det):
                continue
            center_overlap = float((det.get("lane_scores") or {}).get("CENTER", 0.0) or 0.0)
            area_ratio = float(det.get("area_ratio") or 0.0)
            if (
                _is_center_stop_candidate(det)
                and area_ratio >= float(CONFIG["YOLO_OCCUPANCY_STOP_AREA_RATIO"])
            ):
                if center_occupancy_det is None or area_ratio > float(center_occupancy_det.get("area_ratio") or 0.0):
                    center_occupancy_det = det
            for lane, overlap in (det.get("lane_scores") or {}).items():
                if lane not in lane_scores:
                    continue
                try:
                    lane_scores[lane] = max(lane_scores[lane], float(det.get("threat", 0.0)) * float(overlap))
                except Exception:
                    continue
            dist_m = det.get("distance_m")
            if isinstance(dist_m, (int, float)):
                dist_val = float(dist_m)
                nearest_distance = dist_val if nearest_distance is None else min(nearest_distance, dist_val)
        _remember_obstacle_tracks(detections, detect_now)

        obstacle_th = 1.0 / max(float(CONFIG["YOLO_OBSTACLE_DIST_M"]), 0.4)
        bright_frac = _bright_fraction(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        vision_state["last_vision_mode"] = "yolo"
        vision_state["last_bright_frac"] = round(float(bright_frac), 4)
        vision_state["last_obstacle_th_base"] = round(float(obstacle_th), 4)
        vision_state["last_obstacle_th"] = round(float(obstacle_th), 4)
        vision_state["last_left_val"] = round(float(lane_scores["LEFT"]), 4)
        vision_state["last_center_val"] = round(float(lane_scores["CENTER"]), 4)
        vision_state["last_right_val"] = round(float(lane_scores["RIGHT"]), 4)
        vision_state["last_nearest_distance_m"] = round(float(nearest_distance), 2) if nearest_distance is not None else None
        vision_state["last_detections"] = detections
        vision_state["last_detections_held"] = detections_held
        vision_state["last_detection_image_ts"] = float(frame_ts) if frame_ts else 0.0
        vision_state["last_detection_server_ts"] = time.time()
        vision_state["last_lane_markings_visible"] = lane_markings_visible
        if approach_det is not None:
            vision_state["last_approach_hit"] = True
            vision_state["last_approach_reason"] = (
                f"APPROACH:{str(approach_det.get('label', 'obstacle')).upper()}@x"
                f"{float(approach_det.get('approach_ratio') or 1.0):.2f}"
            )
        front_blocking_det = _yolo_front_blocking_detection(
            detections,
            max_distance_m=CONFIG["YOLO_OBSTACLE_DIST_M"],
            require_distance=False,
        )
        emergency_stop_det = None
        if CONFIG["STOP_CONDITION"] == "A":
            emergency_stop_det = _yolo_front_blocking_detection(
                detections,
                max_distance_m=CONFIG["YOLO_EMERGENCY_STOP_M"],
                require_distance=True,
            )
        if center_occupancy_det is not None and front_blocking_det is None:
            front_blocking_det = center_occupancy_det
        if approach_det is not None and emergency_stop_det is None:
            emergency_stop_det = approach_det
        vision_state["last_front_blocked"] = front_blocking_det is not None
        vision_state["last_emergency_stop"] = emergency_stop_det is not None
        vision_state["last_emergency_distance_m"] = (
            round(float(emergency_stop_det["distance_m"]), 2)
            if emergency_stop_det is not None and emergency_stop_det.get("distance_m") is not None
            else None
        )
        vision_state["last_stop_reason"] = ""
        stop_det = emergency_stop_det or front_blocking_det
        if stop_det is not None:
            label = str(stop_det.get("label", "obstacle")).upper()
            dist_m = stop_det.get("distance_m")
            prefix = "EMERGENCY_STOP" if emergency_stop_det is not None else "FRONT_BLOCKED"
            if dist_m is None:
                vision_state["last_stop_reason"] = f"{prefix}:{label}"
            else:
                vision_state["last_stop_reason"] = f"{prefix}:{label}@{float(dist_m):.1f}m"
            if stop_det is approach_det and vision_state["last_approach_reason"]:
                vision_state["last_stop_reason"] = vision_state["last_approach_reason"]
            elif stop_det is center_occupancy_det:
                vision_state["last_stop_reason"] = (
                    f"{prefix}:{label}:AREA@{float(center_occupancy_det.get('area_ratio') or 0.0):.3f}"
                )
        vision_state["detector_ready"] = True
        vision_state["detector_error"] = ""
        vision_state["last_vision_ts"] = time.time()
        if CONFIG["YOLO_HARD_STOP_ON_CENTER"] and emergency_stop_det is not None:
            vision_state["scan_active"] = False
            vision_state["scan_scores"] = {"LEFT": 999.0, "RIGHT": 999.0}
            return "STOP"
        return _decide_turn(lane_scores["CENTER"], lane_scores["LEFT"], lane_scores["RIGHT"], obstacle_th)
    except Exception as exc:
        vision_state["detector_ready"] = False
        vision_state["detector_error"] = str(exc)
        return ""


def _run_vision_pipeline(image_b64: str, frame_ts: Optional[float] = None) -> Tuple[str, str, Optional[float]]:
    if not image_b64:
        return "", "", None
    if CONFIG["VISION_MODE"] == "lite":
        now = time.time()
        if (now - vision_state.get("last_lite_ts", 0.0)) * 1000.0 >= 200:
            vision_state["last_lite_ts"] = now
            lite_cmd = lite_vision_action(image_b64)
            if lite_cmd:
                return lite_cmd, "vision_lite", None
        return "", "", None
    if CONFIG["VISION_MODE"] == "flow":
        now = time.time()
        if (now - vision_state.get("last_flow_ts", 0.0)) * 1000.0 >= CONFIG["FLOW_MIN_INTERVAL_MS"]:
            vision_state["last_flow_ts"] = now
            flow_cmd = flow_vision_action(image_b64)
            if flow_cmd:
                return flow_cmd, "vision_flow", None
        return "", "", None
    if CONFIG["VISION_MODE"] == "yolo":
        now = time.time()
        if (now - vision_state.get("last_yolo_ts", 0.0)) * 1000.0 >= CONFIG["YOLO_MIN_INTERVAL_MS"]:
            vision_state["last_yolo_ts"] = now
            yolo_cmd = yolo_vision_action(image_b64, frame_ts=frame_ts)
            if (
                CONFIG["YOLO_FLOW_FALLBACK"]
                and (not vision_state.get("last_emergency_stop"))
                and (
                    not vision_state.get("detector_ready")
                    or not (vision_state.get("last_detections") or [])
                    or yolo_cmd == "FORWARD"
                )
            ):
                fallback_reason = "no_detections" if not (vision_state.get("last_detections") or []) else "yolo_clear"
                if not vision_state.get("detector_ready"):
                    fallback_reason = "detector_unavailable"
                if (now - vision_state.get("last_flow_ts", 0.0)) * 1000.0 >= CONFIG["FLOW_MIN_INTERVAL_MS"]:
                    vision_state["last_flow_ts"] = now
                    flow_cmd = flow_vision_action(image_b64)
                    if flow_cmd and flow_cmd != "FORWARD":
                        vision_state["last_yolo_fallback_used"] = True
                        vision_state["last_yolo_fallback_reason"] = fallback_reason
                        return flow_cmd, "vision_flow_fallback", None
            if yolo_cmd:
                route = "vision_yolo_emergency_stop" if vision_state.get("last_emergency_stop") else "vision_yolo"
                return yolo_cmd, route, None
        return "", "", None
    if image_b64 and CONFIG["ENABLE_VLM"] and CONFIG["VISION_MODE"] == "vlm":
        now = time.time()
        with metrics_lock:
            last_ts = metrics.get("last_vlm_ts", 0.0)
        if (now - last_ts) * 1000.0 >= CONFIG["VLM_MIN_INTERVAL_MS"]:
            vlm_prompt = (
                "You are an autonomous drone. Analyze this image for immediate obstacles straight ahead. "
                "If the path is perfectly clear, answer 'FORWARD'. If blocked, answer 'LEFT' or 'RIGHT' "
                "to steer away. Output ONLY the ONE English word."
            )
            try:
                ans, vlm_ms = ask_vlm(image_b64, vlm_prompt)
                with metrics_lock:
                    metrics["vlm_calls"] += 1
                    metrics["last_vlm_ts"] = now
                record_latency("vlm", vlm_ms)
                upper = ans.upper()
                for cmd in ["FORWARD", "LEFT", "RIGHT", "STOP", "BACKWARD"]:
                    if cmd in upper:
                        return cmd, "vision_vlm", round(vlm_ms, 1)
            except Exception as exc:
                with metrics_lock:
                    metrics["errors"] += 1
                with state_lock:
                    state["last_error"] = f"VLM error: {exc}"
        return "", "", None
    return "", "", None


def _decide_turn(center_val: float, left_val: float, right_val: float, obstacle_th: float) -> str:
    now = time.time()
    obstacle = center_val >= obstacle_th
    if not obstacle:
        vision_state["scan_active"] = False
        vision_state["scan_scores"] = {"LEFT": 999.0, "RIGHT": 999.0}
        return "FORWARD"

    # Slow scan: rotate left/right and pick the lower obstacle side.
    if CONFIG["VISION_SCAN_ENABLED"]:
        if not vision_state.get("scan_active"):
            vision_state["scan_active"] = True
            vision_state["scan_start_ts"] = now
            vision_state["scan_next_switch_ts"] = now + (CONFIG["VISION_SCAN_STEP_MS"] / 1000.0)
            vision_state["scan_dir"] = "LEFT"
            vision_state["scan_scores"] = {"LEFT": left_val, "RIGHT": right_val}
            return vision_state["scan_dir"]

        # update scores
        vision_state["scan_scores"]["LEFT"] = min(vision_state["scan_scores"]["LEFT"], left_val)
        vision_state["scan_scores"]["RIGHT"] = min(vision_state["scan_scores"]["RIGHT"], right_val)

        if now >= vision_state["scan_start_ts"] + (CONFIG["VISION_SCAN_DURATION_MS"] / 1000.0):
            vision_state["scan_active"] = False
            chosen = "LEFT" if vision_state["scan_scores"]["LEFT"] < vision_state["scan_scores"]["RIGHT"] else "RIGHT"
            vision_state["last_turn"] = chosen
            vision_state["last_turn_ts"] = now
            return chosen

        if now >= vision_state["scan_next_switch_ts"]:
            vision_state["scan_dir"] = "RIGHT" if vision_state["scan_dir"] == "LEFT" else "LEFT"
            vision_state["scan_next_switch_ts"] = now + (CONFIG["VISION_SCAN_STEP_MS"] / 1000.0)

        return vision_state["scan_dir"]

    # Smooth turn selection without scan
    diff = right_val - left_val
    if abs(diff) < CONFIG["VISION_DIFF_TH"]:
        last_turn = vision_state.get("last_turn", "")
        if last_turn and (now - vision_state.get("last_turn_ts", 0.0)) * 1000.0 < CONFIG["VISION_TURN_HOLD_MS"]:
            return last_turn
    turn = "LEFT" if diff > 0 else "RIGHT"
    vision_state["last_turn"] = turn
    vision_state["last_turn_ts"] = now
    return turn


def transcribe_audio(audio_b64: str) -> Tuple[str, float]:
    if not audio_b64 or not whisper_model:
        return "", 0.0
    start = time.perf_counter()
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_audio = f.name
    segments, _ = whisper_model.transcribe(temp_audio)
    transcript = " ".join([s.text for s in segments]).strip()
    try:
        os.remove(temp_audio)
    except OSError:
        pass
    latency_ms = (time.perf_counter() - start) * 1000.0
    return transcript, latency_ms


print("Loading AI Hearing Center (Whisper)...")
whisper_model = None
if CONFIG["ENABLE_WHISPER"] and WhisperModel:
    try:
        whisper_model = WhisperModel(
            CONFIG["WHISPER_MODEL"],
            device=CONFIG["WHISPER_DEVICE"],
            compute_type=CONFIG["WHISPER_COMPUTE"],
        )
    except Exception as exc:
        print("Warning: Whisper failed to load.", exc)
        whisper_model = None
else:
    print("Whisper disabled or not available.")

print("Loading Vision Detector (YOLO)...")
yolo_model = None
yolo_error = ""
if YOLO:
    try:
        yolo_model = YOLO(CONFIG["YOLO_MODEL"])
        vision_state["detector_ready"] = True
    except Exception as exc:
        yolo_error = str(exc)
        vision_state["detector_ready"] = False
        vision_state["detector_error"] = yolo_error
        print("Warning: YOLO failed to load.", exc)
else:
    yolo_error = "ultralytics_not_installed"
    vision_state["detector_ready"] = False
    vision_state["detector_error"] = yolo_error
    print("YOLO disabled or not available.")

print("Loading Vision Depth (Depth Anything V2)...")
depth_processor = None
depth_model = None
depth_error = ""
if CONFIG["DEPTH_ANYTHING_ENABLED"] and AutoImageProcessor and AutoModelForDepthEstimation and torch and Image:
    try:
        depth_processor = AutoImageProcessor.from_pretrained(CONFIG["DEPTH_ANYTHING_MODEL"])
        depth_model = AutoModelForDepthEstimation.from_pretrained(CONFIG["DEPTH_ANYTHING_MODEL"])
        depth_model.eval()
        if CONFIG["DEPTH_ANYTHING_DEVICE"] != "cpu":
            depth_model.to(CONFIG["DEPTH_ANYTHING_DEVICE"])
        vision_state["depth_ready"] = True
    except Exception as exc:
        depth_error = str(exc)
        vision_state["depth_ready"] = False
        vision_state["depth_error"] = depth_error
        print("Warning: Depth Anything failed to load.", exc)
else:
    missing = []
    if not CONFIG["DEPTH_ANYTHING_ENABLED"]:
        missing.append("disabled")
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
        missing.append("transformers_missing")
    if torch is None:
        missing.append("torch_missing")
    if Image is None:
        missing.append("pillow_missing")
    depth_error = ",".join(missing) or "depth_anything_unavailable"
    vision_state["depth_ready"] = False
    vision_state["depth_error"] = depth_error
    print("Depth Anything disabled or not available.")


HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>AirPuff Motherbase</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; background: radial-gradient(circle at top, #172131 0%, #0d1219 48%, #080b10 100%); color: #eee; margin: 0; padding: 10px; }
        h2 { color: #00f0ff; }
        .btn { padding: 20px; margin: 5px; font-size: 16px; font-weight: bold; border-radius: 12px; border: none; cursor: pointer; color: white; touch-action: manipulation; }
        .btn-green { background: #00b894; }
        .btn-red { background: #d63031; }
        .btn-blue { background: #0984e3; }
        img { width: 100%; max-width: 520px; border-radius: 8px; border: 2px solid #555; }
        canvas { width: 100%; max-width: 560px; border-radius: 18px; border: 1px solid #314052; background: #0b0f14; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04); }
        .panel { background: linear-gradient(180deg, rgba(24,31,42,0.94), rgba(11,15,21,0.96)); padding: 15px; border-radius: 16px; margin-bottom: 15px; box-shadow: 0 10px 24px rgba(0,0,0,0.28); border: 1px solid rgba(110,142,180,0.12); }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; max-width: 300px; margin: 0 auto; }
        .log { text-align: left; font-family: monospace; font-size: 14px; color: #00f0ff; background: #000; padding: 10px; border-radius: 5px; height: 120px; overflow-y: auto; }
        .kv { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; text-align: left; font-family: monospace; font-size: 12px; }
        .video-wrap { position: relative; width: 100%; max-width: 520px; margin: 0 auto; border-radius: 12px; overflow: hidden; background: #06090d; border: 1px solid #27313c; }
        #video_feed, #video_overlay { display: block; width: 100%; max-width: none; border: none; border-radius: 0; }
        #video_overlay { position: absolute; inset: 0; height: 100%; background: transparent; pointer-events: none; }
        .sr-head { width: 100%; max-width: 560px; margin: 0 auto 10px; display: flex; justify-content: space-between; align-items: center; color: #d7e9ff; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; }
        .sr-head strong { color: #7fe8ff; font-size: 13px; letter-spacing: 0.18em; }
        .control-strip { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; align-items: center; margin-top: 10px; }
        .btn-slim { padding: 14px 18px; font-size: 14px; }
        .status-chip { display: inline-flex; align-items: center; justify-content: center; min-width: 170px; padding: 12px 16px; border-radius: 999px; border: 1px solid rgba(129, 190, 232, 0.24); background: rgba(12, 25, 38, 0.72); color: #dbefff; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 0.12em; text-transform: uppercase; }
        .status-chip[data-state="on"] { background: rgba(13, 78, 57, 0.62); border-color: rgba(121, 255, 182, 0.34); color: #defff0; }
        .status-chip[data-state="warm"] { background: rgba(82, 58, 14, 0.64); border-color: rgba(255, 214, 111, 0.34); color: #fff4d8; }
        .status-chip[data-state="off"] { background: rgba(44, 26, 26, 0.66); border-color: rgba(255, 150, 150, 0.30); color: #ffe2e2; }
        .status-chip[data-state="busy"] { background: rgba(16, 52, 78, 0.72); border-color: rgba(119, 220, 255, 0.34); color: #e1f7ff; }
        .status-chip[data-state="error"] { background: rgba(72, 22, 22, 0.74); border-color: rgba(255, 120, 120, 0.36); color: #ffe1e1; }
        .video-note { max-width: 520px; margin: 8px auto 0; text-align: left; color: #b8d5f2; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        @media (max-width: 640px) {
            .btn { padding: 16px; }
            .control-strip { flex-direction: column; }
            .status-chip { width: 100%; max-width: 320px; }
        }
    </style>
</head>
<body>
    <h2>AirPuff Terminal</h2>

    <div class="panel">
        <h3 style="margin:5px 0;">Pilot: <span id="mode_label" style="color:#00b894;">AI AUTO</span></h3>
        <div class="control-strip">
            <button class="btn btn-green" onclick="setMode('AUTO')">AI Mode</button>
            <button class="btn btn-red" onclick="setMode('MANUAL')">Manual Mode</button>
        </div>
        <div class="control-strip">
            <div class="status-chip" id="camera_chip" data-state="busy">CAM CHECKING</div>
            <button class="btn btn-blue btn-slim" id="camera_toggle" onclick="toggleCamera()">Camera</button>
            <button class="btn btn-blue btn-slim" id="debug_toggle" onclick="openDebug()">Debug</button>
        </div>
    </div>

    <div class="panel" id="manual_controls" style="display:none;">
        <div class="grid">
            <div></div>
            <button class="btn btn-blue" onmousedown="setCmd('FORWARD')" onmouseup="setCmd('STOP')" ontouchstart="setCmd('FORWARD')" ontouchend="setCmd('STOP')">UP</button>
            <div></div>
            <button class="btn btn-blue" onmousedown="setCmd('LEFT')" onmouseup="setCmd('STOP')" ontouchstart="setCmd('LEFT')" ontouchend="setCmd('STOP')">LEFT</button>
            <button class="btn btn-red" onclick="setCmd('STOP')">STOP</button>
            <button class="btn btn-blue" onmousedown="setCmd('RIGHT')" onmouseup="setCmd('STOP')" ontouchstart="setCmd('RIGHT')" ontouchend="setCmd('STOP')">RIGHT</button>
            <div></div>
            <button class="btn btn-blue" onmousedown="setCmd('BACKWARD')" onmouseup="setCmd('STOP')" ontouchstart="setCmd('BACKWARD')" ontouchend="setCmd('STOP')">DOWN</button>
            <div></div>
        </div>
        <br>
        <button class="btn btn-green" onclick="setCmd('UP')">Alt +</button>
        <button class="btn btn-green" onclick="setCmd('DOWN')">Alt -</button>
    </div>

    <div class="panel">
        <div class="sr-head">
            <strong>SR / YOLO</strong>
            <span>Scene Reconstruction</span>
        </div>
        <canvas id="sr_canvas" width="560" height="360"></canvas>
    </div>

    <div class="panel">
        <div class="video-wrap">
            <img id="video_feed" src="" alt="Awaiting Video Stream..."/>
            <canvas id="video_overlay" width="640" height="480"></canvas>
        </div>
        <div class="video-note" id="camera_note">Camera status pending...</div>
    </div>

    <div class="panel log">
        <div>> Action: <span id="action" style="color:#fdcb6e;">STOP</span></div>
        <div>> User says: <span id="heard" style="color:#fff;">...</span></div>
        <div>> AI Reply: <span id="chat" style="color:#ff7675;">...</span></div>
    </div>

    <div class="panel kv">
        <div>ASR (ms)</div><div id="asr_ms">-</div>
        <div>LLM (ms)</div><div id="llm_ms">-</div>
        <div>VLM (ms)</div><div id="vlm_ms">-</div>
    </div>

    <script>
        function setMode(mode) {
            fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ mode: mode }) });
        }
        function setCmd(cmd) {
            fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ action: cmd }) });
        }
        const srCanvas = document.getElementById('sr_canvas');
        const srCtx = srCanvas ? srCanvas.getContext('2d') : null;
        const videoImg = document.getElementById('video_feed');
        const videoOverlay = document.getElementById('video_overlay');
        const videoCtx = videoOverlay ? videoOverlay.getContext('2d') : null;
        const cameraChip = document.getElementById('camera_chip');
        const cameraToggle = document.getElementById('camera_toggle');
        const cameraNote = document.getElementById('camera_note');
        let cameraBusy = false;
        let lastCameraState = null;
        let lastFrameTs = 0;
        const SR_ASSET_LIBRARY = {
            ground: {
                road: 'PhysicalModel/Ground/Material/MI_RoadColor_01',
                lanePrimary: 'PhysicalModel/Ground/Material/MI_LaneLine_01',
                laneSecondary: 'PhysicalModel/Ground/Material/MI_LaneLine_02',
                grass: 'PhysicalModel/Ground/Material/MI_Grass_01',
                breakRoad: 'PhysicalModel/Ground/Material/MI_BreakRoad_01',
                roadMarking: 'PhysicalModel/Ground/Material/MI_RoadMarking_03',
                water: 'PhysicalModel/Ground/Material/MI_Water_01',
            },
            guide: {
                idle: 'NavigationElements/GuideArrow/SM_GuideArrow_NoCom',
                forward: 'NavigationElements/GuideArrow/SM_GuideArrow_L2',
                left: 'NavigationElements/GuideArrow/SM_GuideArrow_L1',
                right: 'NavigationElements/GuideArrow/SM_GuideArrow_L3',
                changeLane: 'NavigationElements/GuideArrow/SM_GuideArrow_ChangeLane',
                people: 'NavigationElements/GuideForPeople/SM_GuideArrow_01',
            },
            mainCar: {
                model: 'PhysicalModel/MainCar/Model/SM_MainCar_01',
                paint: 'PhysicalModel/MainCar/Material/MI_Paint_01',
                light: 'PhysicalModel/MainCar/Material/MI_Light_01',
                wheel: 'PhysicalModel/MainCar/Material/MI_Wheel_02',
            },
            roadside: {
                guardrail: 'PhysicalModel/Common/Model/SM_SideGuardrail_01',
                pierA: 'PhysicalModel/Common/Model/SM_Pier_01',
                pierB: 'PhysicalModel/Common/Model/SM_Pier_02',
                gantryA: 'PhysicalModel/Common/Model/SM_LongmenFrame_01',
                gantryB: 'PhysicalModel/Common/Model/SM_LongmenFrame_02',
                speedBoard: 'PhysicalModel/Common/Model/SM_SpeedLimitBoard_01',
            },
            building: [
                'PhysicalModel/Building/IntermediateLevelBuilding/Model/M0-1_1_0-1_4_20-40_1_3000_2500_692',
                'PhysicalModel/Building/IntermediateLevelBuilding/Model/M2-5_2_6-3_0_20-40_1_8400_3000_2087',
                'PhysicalModel/Building/IntermediateLevelBuilding/Model/M16-20_3_1_41_1_14400_4500_8153',
            ],
            skyline: [
                { model: 'PhysicalModel/Building/IntermediateLevelBuilding/Model/M0-1_1_0-1_4_20-40_1_3000_2500_692', material: 'PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_01', x: -146, width: 52, height: 70, baseY: 16, tone: '50, 79, 111', alpha: 0.22 },
                { model: 'PhysicalModel/Building/IntermediateLevelBuilding/Model/M2-5_1_4-1_8_20-40_2_4000_2500_1598', material: 'PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_02', x: -92, width: 42, height: 58, baseY: 10, tone: '59, 92, 126', alpha: 0.20 },
                { model: 'PhysicalModel/Building/IntermediateLevelBuilding/Model/M6-10_2_2-2_6_20-40_1_6000_2500_3667', material: 'PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_04', x: 88, width: 40, height: 56, baseY: 10, tone: '59, 92, 126', alpha: 0.20 },
                { model: 'PhysicalModel/Building/IntermediateLevelBuilding/Model/M16-20_3_1_41_1_14400_4500_8153', material: 'PhysicalModel/Building/IntermediateLevelBuilding/Material/MI_BuildingColor_06', x: 142, width: 54, height: 74, baseY: 18, tone: '50, 79, 111', alpha: 0.22 },
            ],
        };
        function syncVideoOverlay() {
            if (!videoOverlay || !videoImg) return;
            const w = videoImg.naturalWidth || videoImg.clientWidth || 640;
            const h = videoImg.naturalHeight || videoImg.clientHeight || 480;
            if (videoOverlay.width !== w || videoOverlay.height !== h) {
                videoOverlay.width = w;
                videoOverlay.height = h;
            }
        }
        if (videoImg) {
            videoImg.addEventListener('load', syncVideoOverlay);
        }
        window.addEventListener('resize', syncVideoOverlay);
        function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
        function lerp(a, b, t) { return a + (b - a) * t; }
        function assetTail(path) {
            const source = (path && typeof path === 'object') ? (path.model || path.material || '') : String(path || '');
            const parts = String(source || '').split('/');
            return parts.length ? parts[parts.length - 1] : '';
        }
        function updateCameraUI(camera) {
            lastCameraState = camera || lastCameraState || {};
            if (!cameraChip || !cameraToggle) return;
            const current = lastCameraState || {};
            const configured = current.configured !== false;
            const reachable = current.reachable !== false;
            const active = !!current.service_active;
            const streaming = !!current.streaming;
            let chipState = 'busy';
            let chipText = 'CAM CHECKING';
            let noteText = 'Camera status pending...';
            let buttonText = 'Turn Camera On';

            if (cameraBusy) {
                chipState = 'busy';
                chipText = active ? 'CAM SWITCHING' : 'CAM STARTING';
                noteText = 'Camera command is being applied on the Raspberry Pi.';
            } else if (!configured) {
                chipState = 'error';
                chipText = 'CAM UNCONFIGURED';
                noteText = 'Camera control is not configured on the server yet.';
            } else if (!reachable && current.last_refresh_ts) {
                chipState = 'error';
                chipText = 'CAM UNREACHABLE';
                noteText = current.last_error || 'The laptop cannot reach the Raspberry Pi camera service right now.';
            } else if (active && streaming) {
                chipState = 'on';
                chipText = `CAM LIVE ${Math.max(0, current.last_frame_age_ms || 0)}MS`;
                noteText = `Pi camera service is running and frames are arriving. Stream age ${Math.max(0, current.last_frame_age_ms || 0)}ms.`;
                buttonText = 'Turn Camera Off';
            } else if (active) {
                chipState = 'warm';
                chipText = 'CAM STARTING';
                noteText = 'Pi camera service is running and waiting for fresh frames to arrive.';
                buttonText = 'Turn Camera Off';
            } else {
                chipState = 'off';
                chipText = 'CAM STANDBY';
                noteText = 'Camera service is stopped. This helps reduce idle power draw and undervoltage risk.';
                buttonText = 'Turn Camera On';
            }

            cameraChip.dataset.state = chipState;
            cameraChip.textContent = chipText;
            cameraToggle.textContent = cameraBusy ? 'Switching...' : buttonText;
            cameraToggle.disabled = cameraBusy || !configured;
            if (cameraNote) cameraNote.textContent = noteText;
        }
        function refreshCameraStatus(force) {
            const refreshFlag = force ? '1' : '0';
            fetch('/api/camera/status?refresh=' + refreshFlag + '&ts=' + Date.now(), { cache: 'no-store' })
                .then(r => r.json())
                .then(data => updateCameraUI(data))
                .catch(() => updateCameraUI(Object.assign({}, lastCameraState || {}, { reachable: false })));
        }
        function toggleCamera() {
            if (cameraBusy) return;
            const current = lastCameraState || {};
            const enable = !current.service_active;
            cameraBusy = true;
            updateCameraUI(current);
            fetch('/api/camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ enabled: enable }),
            })
                .then(r => r.json())
                .then(data => {
                    if (data && data.camera) {
                        updateCameraUI(data.camera);
                    }
                })
                .catch(() => {})
                .finally(() => {
                    cameraBusy = false;
                    setTimeout(() => refreshCameraStatus(true), 450);
                });
        }
        function openDebug() {
            window.open('/debug', '_blank');
        }
        function getDetections(vd) {
            if (vd && vd.detections_stale) {
                return [];
            }
            const detections = Array.isArray(vd && vd.detections) ? [...vd.detections] : [];
            detections.sort((a, b) => {
                const da = typeof a.distance_m === 'number' ? a.distance_m : 999;
                const db = typeof b.distance_m === 'number' ? b.distance_m : 999;
                if (da !== db) return da - db;
                const ta = typeof a.threat === 'number' ? a.threat : 0;
                const tb = typeof b.threat === 'number' ? b.threat : 0;
                if (ta !== tb) return tb - ta;
                return (b.conf || 0) - (a.conf || 0);
            });
            return detections;
        }

        function laneDetection(vd, lane) {
            const detections = getDetections(vd).filter(det => det && det.lane === lane);
            return detections.length ? detections[0] : null;
        }

        function laneState(vd, lane) {
            const detections = getDetections(vd).filter(det => det && det.lane === lane);
            const nearest = detections.find(det => typeof det.distance_m === 'number') || detections[0] || null;
            const threat = detections.reduce((acc, det) => Math.max(acc, typeof det.threat === 'number' ? det.threat : 0), 0);
            const stopLimit = typeof (vd && vd.emergency_stop_limit_m) === 'number' ? vd.emergency_stop_limit_m : 2.0;
            const blocked = nearest && typeof nearest.distance_m === 'number'
                ? nearest.distance_m < stopLimit
                : threat > 0.25;
            return { lane, detections, nearest, threat, blocked };
        }

        function shortLabel(det) {
            const label = String((det && det.label) || 'object').replace(/_/g, ' ');
            const words = label.split(' ').filter(Boolean);
            if (!words.length) return 'OBJECT';
            if (words.length === 1) return words[0].slice(0, 10).toUpperCase();
            return words.slice(0, 2).join(' ').toUpperCase();
        }

        function toneForDetection(det) {
            const kind = det && det.render_kind;
            if (kind === 'tree') {
                return { fill: 'rgba(93, 196, 110, 0.16)', stroke: 'rgba(133, 233, 150, 0.95)', text: '#dffff0' };
            }
            if (kind === 'person') {
                return { fill: 'rgba(255, 177, 119, 0.18)', stroke: 'rgba(255, 204, 150, 0.98)', text: '#fff3e4' };
            }
            if (kind === 'vehicle') {
                return { fill: 'rgba(82, 197, 255, 0.16)', stroke: 'rgba(114, 220, 255, 0.95)', text: '#e6fbff' };
            }
            return { fill: 'rgba(208, 224, 255, 0.12)', stroke: 'rgba(220, 232, 255, 0.92)', text: '#eff5ff' };
        }

        function laneMarkingsVisible(vd) {
            if (vd && typeof vd.lane_markings_visible === 'boolean') return vd.lane_markings_visible;
            return getDetections(vd).some(det => det && det.is_lane_marker);
        }

        function sceneLabelForLane(lane, showLaneMarkings) {
            if (showLaneMarkings) return lane;
            if (lane === 'LEFT') return 'LEFT FIELD';
            if (lane === 'RIGHT') return 'RIGHT FIELD';
            return 'AHEAD';
        }

        function depthFromDetection(det) {
            if (!det) return 0.12;
            if (typeof det.distance_m === 'number') {
                return clamp(1.0 - ((det.distance_m - 0.8) / 10.5), 0.08, 0.97);
            }
            if (Array.isArray(det.box) && det.box.length === 4) {
                return clamp(det.box[3], 0.10, 0.90);
            }
            return 0.28;
        }

        function roundedRectPath(ctx, x, y, w, h, r) {
            const radius = Math.min(r, w / 2, h / 2);
            ctx.beginPath();
            ctx.moveTo(x + radius, y);
            ctx.arcTo(x + w, y, x + w, y + h, radius);
            ctx.arcTo(x + w, y + h, x, y + h, radius);
            ctx.arcTo(x, y + h, x, y, radius);
            ctx.arcTo(x, y, x + w, y, radius);
            ctx.closePath();
        }

        function fillRoundedRect(ctx, x, y, w, h, r) {
            roundedRectPath(ctx, x, y, w, h, r);
            ctx.fill();
        }

        function strokeRoundedRect(ctx, x, y, w, h, r) {
            roundedRectPath(ctx, x, y, w, h, r);
            ctx.stroke();
        }

        function drawPill(ctx, x, y, w, h, fill, stroke, text, textColor) {
            ctx.fillStyle = fill;
            fillRoundedRect(ctx, x, y, w, h, h / 2);
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            strokeRoundedRect(ctx, x, y, w, h, h / 2);
            ctx.fillStyle = textColor;
            ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
            ctx.textAlign = 'center';
            ctx.fillText(text, x + w / 2, y + h / 2 + 4);
            ctx.textAlign = 'left';
        }

        function drawVideoOverlay(vd) {
            if (!videoCtx || !videoOverlay) return;
            syncVideoOverlay();
            const w = videoOverlay.width;
            const h = videoOverlay.height;
            videoCtx.clearRect(0, 0, w, h);
            const detections = getDetections(vd);
            for (const det of detections) {
                if (!det || !Array.isArray(det.box) || det.box.length !== 4) continue;
                const x1 = clamp(det.box[0], 0, 1) * w;
                const y1 = clamp(det.box[1], 0, 1) * h;
                const x2 = clamp(det.box[2], 0, 1) * w;
                const y2 = clamp(det.box[3], 0, 1) * h;
                const bw = Math.max(8, x2 - x1);
                const bh = Math.max(8, y2 - y1);
                const distanceText = typeof det.distance_m === 'number' ? ` ~${det.distance_m.toFixed(1)}m` : '';
                const confText = typeof det.conf === 'number' ? ` ${(det.conf * 100).toFixed(0)}%` : '';
                const label = `${shortLabel(det)}${distanceText}${confText}`;
                const tone = toneForDetection(det);
                const stopLimit = typeof (vd && vd.emergency_stop_limit_m) === 'number' ? vd.emergency_stop_limit_m : 2.0;
                const danger = det.is_obstacle && det.lane === 'CENTER' && typeof det.distance_m === 'number' && det.distance_m <= stopLimit;

                videoCtx.fillStyle = danger ? 'rgba(255, 106, 106, 0.12)' : tone.fill;
                videoCtx.strokeStyle = danger ? 'rgba(255, 130, 130, 0.96)' : tone.stroke;
                videoCtx.lineWidth = danger ? 3 : 2;
                strokeRoundedRect(videoCtx, x1, y1, bw, bh, 8);
                videoCtx.globalAlpha = 0.65;
                fillRoundedRect(videoCtx, x1, y1, bw, bh, 8);
                videoCtx.globalAlpha = 1;

                videoCtx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
                const tw = videoCtx.measureText(label).width + 16;
                const tx = x1;
                const ty = Math.max(5, y1 - 22);
                videoCtx.fillStyle = 'rgba(6, 15, 24, 0.92)';
                fillRoundedRect(videoCtx, tx, ty, tw, 18, 9);
                videoCtx.strokeStyle = danger ? 'rgba(255, 140, 140, 0.88)' : 'rgba(129, 190, 232, 0.45)';
                videoCtx.lineWidth = 1;
                strokeRoundedRect(videoCtx, tx, ty, tw, 18, 9);
                videoCtx.fillStyle = danger ? 'rgba(255, 226, 226, 0.98)' : tone.text;
                videoCtx.fillText(label, tx + 8, ty + 13);
            }
        }

        function drawSR(vd, action, mode) {
            if (!srCtx || !srCanvas) return;
            const ctx = srCtx;
            const w = srCanvas.width;
            const h = srCanvas.height;
            const detections = getDetections(vd).slice(0, 8);
            const lanes = ['LEFT', 'CENTER', 'RIGHT'];
            const laneStates = lanes.map(lane => laneState(vd, lane));
            const nearest = detections.find(det => typeof det.distance_m === 'number') || detections[0] || null;
            const detectorReady = !(vd && vd.detector_ready === false);
            const emergencyStop = !!(vd && vd.emergency_stop);
            const streaming = !!(vd && vd.streaming);
            const showLaneMarkings = laneMarkingsVisible(vd);
            const suggestedAction = String((vd && vd.suggested_action) || '').toUpperCase();
            const manualDisplayAction = (mode === 'MANUAL' && action === 'STOP' && ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'].includes(suggestedAction))
                ? suggestedAction
                : action;
            const sceneGroundAsset = showLaneMarkings ? SR_ASSET_LIBRARY.ground.roadMarking : SR_ASSET_LIBRARY.ground.breakRoad;
            let guideAsset = SR_ASSET_LIBRARY.guide.idle;
            if (mode === 'AUTO') {
                guideAsset = action === 'LEFT'
                    ? SR_ASSET_LIBRARY.guide.left
                    : action === 'RIGHT'
                        ? SR_ASSET_LIBRARY.guide.right
                        : action === 'FORWARD'
                            ? SR_ASSET_LIBRARY.guide.forward
                            : SR_ASSET_LIBRARY.guide.idle;
            } else {
                guideAsset = manualDisplayAction === 'LEFT'
                    ? SR_ASSET_LIBRARY.guide.left
                    : manualDisplayAction === 'RIGHT'
                        ? SR_ASSET_LIBRARY.guide.right
                        : manualDisplayAction === 'FORWARD'
                            ? SR_ASSET_LIBRARY.guide.forward
                            : suggestedAction && suggestedAction !== 'STOP'
                                ? SR_ASSET_LIBRARY.guide.people
                                : SR_ASSET_LIBRARY.guide.idle;
            }
            const stopLimit = typeof (vd && vd.emergency_stop_limit_m) === 'number' ? vd.emergency_stop_limit_m : 2.0;
            const nearestText = emergencyStop && typeof (vd && vd.emergency_stop_distance_m) === 'number'
                ? `STOP ${vd.emergency_stop_distance_m.toFixed(1)}m`
                : nearest && typeof nearest.distance_m === 'number'
                    ? `${nearest.distance_m.toFixed(1)}m`
                    : (detections.length ? 'tracking' : 'clear');

            ctx.clearRect(0, 0, w, h);

            const sky = ctx.createLinearGradient(0, 0, 0, h);
            sky.addColorStop(0, '#07111a');
            sky.addColorStop(0.45, '#0c1823');
            sky.addColorStop(0.72, '#0a0f16');
            sky.addColorStop(1, '#05070b');
            ctx.fillStyle = sky;
            ctx.fillRect(0, 0, w, h);

            const horizonGlow = ctx.createRadialGradient(w * 0.5, h * 0.26, 10, w * 0.5, h * 0.30, w * 0.45);
            horizonGlow.addColorStop(0, 'rgba(84, 187, 255, 0.22)');
            horizonGlow.addColorStop(0.45, 'rgba(38, 92, 166, 0.10)');
            horizonGlow.addColorStop(1, 'rgba(10, 16, 24, 0)');
            ctx.fillStyle = horizonGlow;
            ctx.fillRect(0, 0, w, h);

            const roadTopY = h * 0.20;
            const roadBotY = h * 0.97;
            const roadTopW = w * 0.28;
            const roadBotW = w * 0.95;
            const cx = w / 2;

            function roadHalfWidthAt(y) {
                const t = clamp((y - roadTopY) / (roadBotY - roadTopY), 0, 1);
                return lerp(roadTopW, roadBotW, t) / 2;
            }

            function edgeXAt(y, side) {
                return cx + (side < 0 ? -1 : 1) * roadHalfWidthAt(y);
            }

            function laneBoundXAt(edgeIdx, y) {
                const left = edgeXAt(y, -1);
                const right = edgeXAt(y, 1);
                return lerp(left, right, edgeIdx / 3);
            }

            function laneXAtY(laneIdx, y) {
                return lerp(laneBoundXAt(laneIdx, y), laneBoundXAt(laneIdx + 1, y), 0.5);
            }

            const roadSurface = ctx.createLinearGradient(0, roadTopY, 0, roadBotY);
            if (showLaneMarkings) {
                roadSurface.addColorStop(0, '#0e151e');
                roadSurface.addColorStop(1, '#13212d');
            } else {
                roadSurface.addColorStop(0, '#10222c');
                roadSurface.addColorStop(0.58, '#143344');
                roadSurface.addColorStop(1, '#173947');
            }
            ctx.fillStyle = roadSurface;
            ctx.beginPath();
            ctx.moveTo(cx - roadTopW / 2, roadTopY);
            ctx.lineTo(cx + roadTopW / 2, roadTopY);
            ctx.lineTo(cx + roadBotW / 2, roadBotY);
            ctx.lineTo(cx - roadBotW / 2, roadBotY);
            ctx.closePath();
            ctx.fill();

            const shoulder = ctx.createLinearGradient(0, roadTopY, 0, roadBotY);
            if (showLaneMarkings) {
                shoulder.addColorStop(0, 'rgba(40, 62, 82, 0.10)');
                shoulder.addColorStop(1, 'rgba(78, 114, 150, 0.26)');
            } else {
                shoulder.addColorStop(0, 'rgba(45, 86, 67, 0.16)');
                shoulder.addColorStop(1, 'rgba(92, 142, 103, 0.30)');
            }
            ctx.fillStyle = shoulder;
            ctx.beginPath();
            ctx.moveTo(0, roadBotY);
            ctx.lineTo(cx - roadBotW / 2, roadBotY);
            ctx.lineTo(cx - roadTopW / 2, roadTopY);
            ctx.lineTo(0, roadTopY + 14);
            ctx.closePath();
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(w, roadBotY);
            ctx.lineTo(cx + roadBotW / 2, roadBotY);
            ctx.lineTo(cx + roadTopW / 2, roadTopY);
            ctx.lineTo(w, roadTopY + 14);
            ctx.closePath();
            ctx.fill();

            if (showLaneMarkings) {
                laneStates.forEach((state, idx) => {
                    const laneTopLeft = laneBoundXAt(idx, roadTopY + 2);
                    const laneTopRight = laneBoundXAt(idx + 1, roadTopY + 2);
                    const laneBotLeft = laneBoundXAt(idx, roadBotY - 2);
                    const laneBotRight = laneBoundXAt(idx + 1, roadBotY - 2);
                    const alpha = state.blocked ? 0.22 : clamp(state.threat * 1.25, 0.04, 0.14);
                    const fill = state.blocked
                        ? `rgba(255, 112, 112, ${alpha})`
                        : `rgba(89, 196, 255, ${alpha})`;
                    ctx.fillStyle = fill;
                    ctx.beginPath();
                    ctx.moveTo(laneTopLeft, roadTopY + 2);
                    ctx.lineTo(laneTopRight, roadTopY + 2);
                    ctx.lineTo(laneBotRight, roadBotY - 2);
                    ctx.lineTo(laneBotLeft, roadBotY - 2);
                    ctx.closePath();
                    ctx.fill();
                });

                ctx.strokeStyle = 'rgba(255, 166, 98, 0.24)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(edgeXAt(roadTopY, -1), roadTopY);
                ctx.lineTo(edgeXAt(roadBotY, -1), roadBotY);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(edgeXAt(roadTopY, 1), roadTopY);
                ctx.lineTo(edgeXAt(roadBotY, 1), roadBotY);
                ctx.stroke();

                ctx.setLineDash([10, 12]);
                ctx.strokeStyle = 'rgba(236, 245, 255, 0.32)';
                for (let laneIdx = 1; laneIdx <= 2; laneIdx++) {
                    ctx.beginPath();
                    ctx.moveTo(laneBoundXAt(laneIdx, roadTopY + 8), roadTopY + 8);
                    ctx.lineTo(laneBoundXAt(laneIdx, roadBotY - 8), roadBotY - 8);
                    ctx.stroke();
                }
                ctx.setLineDash([]);
            } else {
                const freeGroundGlow = ctx.createRadialGradient(cx, roadBotY - 28, 20, cx, roadBotY - 34, roadBotW * 0.54);
                freeGroundGlow.addColorStop(0, emergencyStop ? 'rgba(255, 126, 126, 0.14)' : 'rgba(111, 223, 255, 0.14)');
                freeGroundGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');
                ctx.fillStyle = freeGroundGlow;
                ctx.beginPath();
                ctx.moveTo(cx - roadTopW / 2, roadTopY);
                ctx.lineTo(cx + roadTopW / 2, roadTopY);
                ctx.lineTo(cx + roadBotW / 2, roadBotY);
                ctx.lineTo(cx - roadBotW / 2, roadBotY);
                ctx.closePath();
                ctx.fill();
            }

            ctx.strokeStyle = 'rgba(118, 180, 230, 0.10)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 5; i++) {
                const y = lerp(roadTopY + 10, roadBotY - 12, i / 6);
                ctx.beginPath();
                ctx.moveTo(edgeXAt(y, -1), y);
                ctx.lineTo(edgeXAt(y, 1), y);
                ctx.stroke();
            }

            function drawBuildingBlock(x, baseY, bw, bh, tone, alpha) {
                ctx.fillStyle = `rgba(${tone}, ${alpha})`;
                fillRoundedRect(ctx, x - bw / 2, baseY - bh, bw, bh, Math.min(8, bw * 0.08));
                ctx.strokeStyle = `rgba(129, 198, 255, ${alpha * 0.75})`;
                ctx.lineWidth = 1;
                strokeRoundedRect(ctx, x - bw / 2, baseY - bh, bw, bh, Math.min(8, bw * 0.08));
                ctx.strokeStyle = `rgba(117, 205, 255, ${alpha * 0.35})`;
                for (let row = 0; row < 3; row++) {
                    const yy = baseY - bh + 12 + row * (bh / 3.4);
                    ctx.beginPath();
                    ctx.moveTo(x - bw * 0.34, yy);
                    ctx.lineTo(x + bw * 0.34, yy);
                    ctx.stroke();
                }
            }

            function drawGuardrail(side, y, scale) {
                const x = edgeXAt(y, side) + side * 18 * scale;
                const railLen = 42 * scale;
                ctx.strokeStyle = 'rgba(154, 190, 220, 0.44)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x - railLen / 2, y);
                ctx.lineTo(x + railLen / 2, y);
                ctx.stroke();
                ctx.strokeStyle = 'rgba(83, 109, 128, 0.72)';
                ctx.beginPath();
                ctx.moveTo(x - railLen / 2, y + 5 * scale);
                ctx.lineTo(x + railLen / 2, y + 5 * scale);
                ctx.stroke();
                for (let i = -1; i <= 1; i++) {
                    ctx.strokeStyle = 'rgba(125, 154, 182, 0.56)';
                    ctx.beginPath();
                    ctx.moveTo(x + i * railLen * 0.28, y - 2 * scale);
                    ctx.lineTo(x + i * railLen * 0.28, y + 8 * scale);
                    ctx.stroke();
                }
            }

            function drawPier(side, y, scale, variant) {
                const x = edgeXAt(y, side) + side * 42 * scale;
                const pierW = variant === 'B' ? 18 * scale : 14 * scale;
                const pierH = variant === 'B' ? 38 * scale : 30 * scale;
                drawShadow(x, y + 12 * scale, pierW * 0.85, 5 * scale, 0.18);
                ctx.fillStyle = variant === 'B' ? 'rgba(125, 144, 164, 0.80)' : 'rgba(110, 130, 150, 0.76)';
                fillRoundedRect(ctx, x - pierW / 2, y - pierH, pierW, pierH, 5 * scale);
                ctx.fillStyle = 'rgba(164, 188, 209, 0.42)';
                fillRoundedRect(ctx, x - pierW * 0.22, y - pierH + 4 * scale, pierW * 0.44, pierH - 8 * scale, 3 * scale);
            }

            function drawGantry(y, variant) {
                const left = edgeXAt(y, -1) - 16;
                const right = edgeXAt(y, 1) + 16;
                const height = variant === 'B' ? 42 : 36;
                ctx.strokeStyle = variant === 'B' ? 'rgba(126, 160, 191, 0.55)' : 'rgba(104, 136, 166, 0.52)';
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.moveTo(left, y);
                ctx.lineTo(left, y - height);
                ctx.lineTo(right, y - height);
                ctx.lineTo(right, y);
                ctx.stroke();
                ctx.fillStyle = 'rgba(28, 45, 61, 0.72)';
                fillRoundedRect(ctx, cx - 54, y - height - 4, 108, 18, 8);
            }

            function drawSpeedBoard(y, text) {
                const x = edgeXAt(y, 1) + 36;
                drawShadow(x, y + 10, 12, 4, 0.18);
                ctx.strokeStyle = 'rgba(230, 240, 251, 0.82)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y - 14, 14, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
                ctx.beginPath();
                ctx.arc(x, y - 14, 12, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = 'rgba(248, 120, 120, 0.92)';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(x, y - 14, 12, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = '#18222d';
                ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, monospace';
                ctx.textAlign = 'center';
                ctx.fillText(text, x, y - 10);
                ctx.textAlign = 'left';
                ctx.strokeStyle = 'rgba(150, 175, 198, 0.70)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x, y + 16);
                ctx.stroke();
            }

            function drawShadow(x, y, rx, ry, alpha) {
                ctx.fillStyle = `rgba(0,0,0,${alpha})`;
                ctx.beginPath();
                ctx.ellipse(x, y + 5, rx, ry, 0, 0, Math.PI * 2);
                ctx.fill();
            }

            function drawTree(x, y, s) {
                drawShadow(x, y, 13 * s, 5 * s, 0.30);
                ctx.fillStyle = '#355f3d';
                ctx.beginPath();
                ctx.arc(x, y - 10 * s, 11 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#467a4f';
                ctx.beginPath();
                ctx.arc(x + 6 * s, y - 13 * s, 8 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#7b5733';
                ctx.fillRect(x - 2 * s, y - 2 * s, 4 * s, 11 * s);
            }

            function drawPerson(x, y, s) {
                drawShadow(x, y, 9 * s, 4 * s, 0.26);
                ctx.fillStyle = '#f0f7ff';
                ctx.beginPath();
                ctx.arc(x, y - 10 * s, 4 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#59b5ff';
                ctx.fillRect(x - 3 * s, y - 6 * s, 6 * s, 12 * s);
                ctx.fillStyle = '#bac7d3';
                ctx.fillRect(x - 6 * s, y + 6 * s, 4 * s, 8 * s);
                ctx.fillRect(x + 2 * s, y + 6 * s, 4 * s, 8 * s);
            }

            function drawSeat(x, y, s) {
                drawShadow(x, y, 11 * s, 4 * s, 0.28);
                ctx.fillStyle = '#87939a';
                ctx.fillRect(x - 8 * s, y - 6 * s, 16 * s, 8 * s);
                ctx.fillStyle = '#9baab0';
                ctx.fillRect(x - 8 * s, y - 16 * s, 16 * s, 10 * s);
                ctx.fillStyle = '#2d3438';
                ctx.fillRect(x - 6 * s, y + 2 * s, 12 * s, 3 * s);
            }

            function drawVehicle(x, y, s, accent) {
                drawShadow(x, y, 18 * s, 5 * s, 0.32);
                ctx.fillStyle = '#f2f8ff';
                fillRoundedRect(ctx, x - 14 * s, y - 10 * s, 28 * s, 12 * s, 4 * s);
                ctx.fillStyle = accent || '#74d2ff';
                fillRoundedRect(ctx, x - 8 * s, y - 17 * s, 16 * s, 9 * s, 4 * s);
                ctx.fillStyle = '#223341';
                fillRoundedRect(ctx, x - 9 * s, y + 1 * s, 6 * s, 4 * s, 2 * s);
                fillRoundedRect(ctx, x + 3 * s, y + 1 * s, 6 * s, 4 * s, 2 * s);
                ctx.fillStyle = 'rgba(119, 220, 255, 0.90)';
                fillRoundedRect(ctx, x - 12 * s, y - 6 * s, 4 * s, 2 * s, 1 * s);
                fillRoundedRect(ctx, x + 8 * s, y - 6 * s, 4 * s, 2 * s, 1 * s);
            }

            function drawGuideArrowGlyph(x, y, scale, variant, alpha) {
                const headColor = emergencyStop ? `rgba(255, 146, 146, ${alpha})` : `rgba(118, 222, 255, ${alpha})`;
                const shaftColor = emergencyStop ? `rgba(255, 103, 103, ${alpha * 0.75})` : `rgba(70, 172, 255, ${alpha * 0.72})`;
                ctx.fillStyle = shaftColor;
                fillRoundedRect(ctx, x - 4 * scale, y - 18 * scale, 8 * scale, 28 * scale, 4 * scale);
                ctx.fillStyle = headColor;
                ctx.beginPath();
                if (variant === 'left') {
                    ctx.moveTo(x - 16 * scale, y - 8 * scale);
                    ctx.lineTo(x + 6 * scale, y - 18 * scale);
                    ctx.lineTo(x + 6 * scale, y + 2 * scale);
                } else if (variant === 'right') {
                    ctx.moveTo(x + 16 * scale, y - 8 * scale);
                    ctx.lineTo(x - 6 * scale, y - 18 * scale);
                    ctx.lineTo(x - 6 * scale, y + 2 * scale);
                } else {
                    ctx.moveTo(x, y - 22 * scale);
                    ctx.lineTo(x - 14 * scale, y - 4 * scale);
                    ctx.lineTo(x + 14 * scale, y - 4 * scale);
                }
                ctx.closePath();
                ctx.fill();
            }

            const skylineY = roadTopY + 34;
            (SR_ASSET_LIBRARY.skyline || []).forEach(asset => {
                drawBuildingBlock(
                    cx + (asset.x || 0),
                    skylineY + (asset.baseY || 0),
                    asset.width || 48,
                    asset.height || 64,
                    asset.tone || '59, 92, 126',
                    asset.alpha || 0.20,
                );
            });
            if (showLaneMarkings) {
                drawPier(-1, roadTopY + 68, 0.95, 'A');
                drawPier(1, roadTopY + 68, 0.95, 'B');
                drawGantry(roadTopY + 72, emergencyStop ? 'B' : 'A');
                for (let i = 0; i < 5; i++) {
                    const y = lerp(roadTopY + 60, roadBotY - 32, i / 4);
                    const scale = lerp(0.38, 1.0, i / 4);
                    drawGuardrail(-1, y, scale);
                    drawGuardrail(1, y, scale);
                }
                if (nearest && typeof nearest.distance_m === 'number') {
                    const boardText = emergencyStop ? 'STOP' : String(Math.max(10, Math.min(99, Math.round(nearest.distance_m * 10))));
                    drawSpeedBoard(roadTopY + 118, boardText);
                }
            } else {
                const groveY = [roadTopY + 94, roadTopY + 142, roadTopY + 194];
                groveY.forEach((y, idx) => {
                    const scale = 0.42 + idx * 0.18;
                    drawTree(edgeXAt(y, -1) - 20 * scale, y + 10 * scale, scale);
                    drawTree(edgeXAt(y, 1) + 20 * scale, y + 10 * scale, scale * 1.04);
                });
            }

            if (mode === 'AUTO') {
                let targetLane = 1;
                if (action === 'LEFT') targetLane = 0;
                if (action === 'RIGHT') targetLane = 2;
                const targetX = laneXAtY(targetLane, roadTopY + 20);
                const baseX = cx;
                const pathWidthTop = roadTopW * 0.12;
                const pathWidthBot = roadBotW * 0.16;
                const bendY = h * 0.58;
                const routeFill = action === 'STOP' ? 'rgba(255, 129, 129, 0.18)' : 'rgba(42, 170, 255, 0.38)';
                const routeStroke = action === 'STOP' ? 'rgba(255, 167, 167, 0.80)' : 'rgba(144, 217, 255, 0.92)';
                ctx.fillStyle = routeFill;
                ctx.beginPath();
                ctx.moveTo(baseX - pathWidthBot / 2, roadBotY);
                ctx.quadraticCurveTo(cx, bendY, targetX - pathWidthTop / 2, roadTopY + 16);
                ctx.lineTo(targetX + pathWidthTop / 2, roadTopY + 16);
                ctx.quadraticCurveTo(cx, bendY, baseX + pathWidthBot / 2, roadBotY);
                ctx.closePath();
                ctx.fill();
                ctx.strokeStyle = routeStroke;
                ctx.lineWidth = 2;
                ctx.stroke();
                const arrowVariant = action === 'LEFT' ? 'left' : action === 'RIGHT' ? 'right' : 'forward';
                const arrowXs = [0.22, 0.46, 0.70].map(t => lerp(baseX, targetX, t));
                const arrowYs = [0.22, 0.46, 0.70].map(t => lerp(roadBotY - 24, roadTopY + 42, t));
                arrowXs.forEach((x, idx) => drawGuideArrowGlyph(x, arrowYs[idx], 0.74 - idx * 0.08, arrowVariant, emergencyStop ? 0.84 : 0.70));
            } else if (mode === 'MANUAL' && manualDisplayAction && manualDisplayAction !== 'STOP') {
                const suggestedOnly = action === 'STOP' && manualDisplayAction !== 'STOP';
                ctx.strokeStyle = suggestedOnly ? 'rgba(117, 210, 255, 0.42)' : 'rgba(117, 210, 255, 0.75)';
                ctx.lineWidth = 2;
                ctx.setLineDash([8, 8]);
                if (manualDisplayAction === 'LEFT' || manualDisplayAction === 'RIGHT') {
                    const dir = manualDisplayAction === 'LEFT' ? -1 : 1;
                    ctx.beginPath();
                    ctx.moveTo(cx, roadBotY - 34);
                    ctx.quadraticCurveTo(cx + dir * 42, roadBotY - 50, cx + dir * 66, roadBotY - 28);
                    ctx.stroke();
                } else if (manualDisplayAction === 'FORWARD') {
                    ctx.beginPath();
                    ctx.moveTo(cx, roadBotY - 34);
                    ctx.lineTo(cx, roadBotY - 78);
                    ctx.stroke();
                } else if (manualDisplayAction === 'BACKWARD') {
                    ctx.beginPath();
                    ctx.moveTo(cx, roadBotY - 16);
                    ctx.lineTo(cx, roadBotY + 10);
                    ctx.stroke();
                }
                ctx.setLineDash([]);
                const manualVariant = manualDisplayAction === 'LEFT' ? 'left' : manualDisplayAction === 'RIGHT' ? 'right' : 'forward';
                drawGuideArrowGlyph(cx, roadBotY - 46, suggestedOnly ? 0.76 : 0.88, manualVariant, suggestedOnly ? 0.42 : 0.72);
            } else if (mode === 'MANUAL') {
                const idleVariant = suggestedAction === 'LEFT' ? 'left' : suggestedAction === 'RIGHT' ? 'right' : 'forward';
                drawGuideArrowGlyph(cx, roadBotY - 44, 0.72, idleVariant, suggestedAction && suggestedAction !== 'STOP' ? 0.28 : 0.24);
            }

            const sceneDetections = [...detections].sort((a, b) => depthFromDetection(a) - depthFromDetection(b));
            for (const det of sceneDetections) {
                const depth = depthFromDetection(det);
                const laneIdx = det.lane === 'LEFT' ? 0 : det.lane === 'RIGHT' ? 2 : 1;
                const y = lerp(roadTopY + 24, roadBotY - 76, depth);
                const rawCenter = Array.isArray(det.box) && det.box.length === 4 ? (det.box[0] + det.box[2]) / 2 : 0.5;
                const laneCenter = laneXAtY(laneIdx, y);
                const halfWidth = roadHalfWidthAt(y);
                const roadX = cx + (rawCenter - 0.5) * halfWidth * 1.45;
                const laneSnapWeight = showLaneMarkings ? 0.38 : 0.82;
                const x = clamp(lerp(laneCenter, roadX, laneSnapWeight), cx - halfWidth * 0.96, cx + halfWidth * 0.96);
                const boxHeight = Array.isArray(det.box) && det.box.length === 4 ? Math.max(0.04, det.box[3] - det.box[1]) : 0.12;
                const scale = 0.42 + depth * 1.08 + boxHeight * 1.4;
                const tone = toneForDetection(det);
                const emergencyDet = emergencyStop && det.lane === 'CENTER' && typeof det.distance_m === 'number' && det.distance_m <= stopLimit;

                ctx.strokeStyle = det.lane === 'CENTER' && det.is_obstacle ? 'rgba(255, 132, 132, 0.36)' : 'rgba(108, 200, 255, 0.20)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(cx, roadBotY - 40);
                ctx.lineTo(x, y + 8);
                ctx.stroke();

                if (det.render_kind === 'tree') drawTree(x, y, scale);
                if (det.render_kind === 'person') drawPerson(x, y, scale);
                if (det.render_kind === 'seat') drawSeat(x, y, scale);
                if (det.render_kind === 'vehicle') drawVehicle(x, y, scale, emergencyDet ? '#ff9d9d' : '#75d5ff');
                if (!det.render_kind) drawSeat(x, y, scale);

                const meta = typeof det.distance_m === 'number' ? `${det.distance_m.toFixed(1)}m` : 'TRACK';
                const label = `${shortLabel(det)}  ${meta}`;
                ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
                const labelWidth = ctx.measureText(label).width + 14;
                const labelX = clamp(x - labelWidth / 2, 10, w - labelWidth - 10);
                const labelY = clamp(y - 24 - scale * 8, 44, roadBotY - 88);
                ctx.fillStyle = 'rgba(8, 18, 28, 0.92)';
                fillRoundedRect(ctx, labelX, labelY, labelWidth, 18, 9);
                ctx.strokeStyle = tone.stroke;
                ctx.lineWidth = 1;
                strokeRoundedRect(ctx, labelX, labelY, labelWidth, 18, 9);
                ctx.fillStyle = tone.text;
                ctx.fillText(label, labelX + 7, labelY + 13);
            }

            const selfY = roadBotY - 18;
            const carW = roadBotW * 0.13;
            const carH = carW * 0.56;
            drawShadow(cx, selfY, carW * 0.45, carH * 0.24, 0.45);
            ctx.fillStyle = '#edf6ff';
            fillRoundedRect(ctx, cx - carW / 2, selfY - carH, carW, carH, 10);
            ctx.fillStyle = emergencyStop ? '#ff9e9e' : '#77d5ff';
            fillRoundedRect(ctx, cx - carW / 3, selfY - carH * 0.82, carW * 0.66, carH * 0.46, 8);
            ctx.fillStyle = '#213240';
            fillRoundedRect(ctx, cx - carW / 2 + 6, selfY - 4, 8, 5, 2);
            fillRoundedRect(ctx, cx + carW / 2 - 14, selfY - 4, 8, 5, 2);
            ctx.fillStyle = emergencyStop ? 'rgba(255, 204, 204, 0.92)' : 'rgba(96, 212, 255, 0.82)';
            fillRoundedRect(ctx, cx - carW / 2 + 10, selfY - 2, 7, 3, 1);
            fillRoundedRect(ctx, cx + carW / 2 - 17, selfY - 2, 7, 3, 1);

            const headerState = !detectorReady ? 'YOLO OFFLINE' : streaming ? 'YOLO HD LIVE' : 'CAM STANDBY';
            const headerFill = !detectorReady ? 'rgba(120, 35, 35, 0.24)' : streaming ? 'rgba(30, 109, 153, 0.24)' : 'rgba(92, 64, 24, 0.28)';
            const headerStroke = !detectorReady ? 'rgba(255, 134, 134, 0.56)' : streaming ? 'rgba(105, 214, 255, 0.56)' : 'rgba(255, 210, 132, 0.40)';
            drawPill(ctx, 16, 16, 120, 22, headerFill, headerStroke, headerState, '#eaf8ff');
            drawPill(ctx, w / 2 - 60, 16, 120, 22, 'rgba(14, 26, 38, 0.72)', 'rgba(145, 189, 231, 0.22)', `${detections.length} TARGETS`, '#d8e7fb');
            drawPill(ctx, w - 150, 16, 134, 22, emergencyStop ? 'rgba(102, 28, 28, 0.80)' : 'rgba(14, 26, 38, 0.72)', emergencyStop ? 'rgba(255, 156, 156, 0.34)' : 'rgba(145, 189, 231, 0.22)', `NEAREST ${nearestText}`, '#d8e7fb');
            drawPill(ctx, 18, h - 84, 136, 20, 'rgba(9, 22, 34, 0.72)', 'rgba(124, 200, 255, 0.22)', `HD ${assetTail(SR_ASSET_LIBRARY.mainCar.model)}`, '#d9ecff');
            drawPill(ctx, 164, h - 84, 168, 20, 'rgba(9, 22, 34, 0.72)', 'rgba(124, 200, 255, 0.22)', `NAV ${assetTail(guideAsset)}`, '#d9ecff');
            drawPill(ctx, 342, h - 84, 200, 20, 'rgba(9, 22, 34, 0.72)', 'rgba(124, 200, 255, 0.22)', `SCENE ${assetTail(sceneGroundAsset)}`, '#d9ecff');

            if (!detections.length && detectorReady && streaming) {
                ctx.strokeStyle = 'rgba(109, 194, 255, 0.28)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([7, 8]);
                ctx.beginPath();
                ctx.arc(cx, roadBotY - 20, 64, Math.PI * 1.06, Math.PI * 1.94);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(cx, roadBotY - 20, 112, Math.PI * 1.10, Math.PI * 1.90);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(210, 234, 255, 0.86)';
                ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
                const clearText = showLaneMarkings ? 'YOLO CLEAR VIEW' : 'OPEN GROUND';
                ctx.fillText(clearText, w / 2 - Math.min(52, ctx.measureText(clearText).width / 2), 56);
            } else if (!streaming) {
                ctx.fillStyle = 'rgba(255, 228, 196, 0.90)';
                ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
                ctx.fillText('CAMERA STANDBY', w / 2 - 48, 56);
            }
            if (emergencyStop) {
                ctx.fillStyle = 'rgba(255, 214, 214, 0.94)';
                ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, monospace';
                const reasonText = String((vd && vd.stop_reason) || 'EMERGENCY STOP').slice(0, 38);
                ctx.fillText(reasonText, w / 2 - Math.min(102, ctx.measureText(reasonText).width / 2), 78);
            }

            laneStates.forEach((state, idx) => {
                const cardW = 116;
                const gap = 10;
                const startX = w / 2 - (cardW * 3 + gap * 2) / 2;
                const x = startX + idx * (cardW + gap);
                const y = h - 40;
                const sceneLabel = sceneLabelForLane(state.lane, showLaneMarkings);
                const text = state.nearest
                    ? `${sceneLabel} ${typeof state.nearest.distance_m === 'number' ? state.nearest.distance_m.toFixed(1) + 'm' : shortLabel(state.nearest)}`
                    : `${sceneLabel} CLEAR`;
                const fill = state.blocked ? 'rgba(95, 28, 28, 0.70)' : 'rgba(10, 24, 37, 0.74)';
                const stroke = state.blocked ? 'rgba(255, 142, 142, 0.44)' : 'rgba(120, 196, 255, 0.28)';
                drawPill(ctx, x, y, cardW, 22, fill, stroke, text, '#eef7ff');
            });

            if (vd && vd.scan_active) {
                ctx.fillStyle = 'rgba(145, 231, 255, 0.88)';
                ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
                ctx.fillText(`SCAN ${vd.scan_dir || ''}`, 18, 52);
            }

            ctx.fillStyle = 'rgba(216, 233, 255, 0.82)';
            ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
            ctx.fillText(`MODE ${mode || 'AUTO'}`, 18, h - 58);
            ctx.fillText(`CMD ${action || 'STOP'}`, w - 104, h - 58);
        }
        let stateBusy = false;
        refreshCameraStatus(true);
        setInterval(() => refreshCameraStatus(false), 2500);
        const statePollMs = 100;
        setInterval(() => {
            if (stateBusy) return;
            stateBusy = true;
            fetch('/api/state?ts=' + Date.now() + '&image_ts=' + encodeURIComponent(lastFrameTs || 0), { cache: 'no-store' }).then(r => r.json()).then(data => {
                document.getElementById('mode_label').innerText = data.mode;
                document.getElementById('mode_label').style.color = data.mode === 'AUTO' ? '#00b894' : '#d63031';
                document.getElementById('action').innerText = data.latest_action;
                document.getElementById('heard').innerText = data.latest_transcript;
                document.getElementById('chat').innerText = data.latest_chat;
                document.getElementById('asr_ms').innerText = data.last_asr_ms ?? '-';
                document.getElementById('llm_ms').innerText = data.last_llm_ms ?? '-';
                document.getElementById('vlm_ms').innerText = data.last_vlm_ms ?? '-';
                if (data.latest_image && data.latest_image_ts && data.latest_image_ts !== lastFrameTs) {
                    lastFrameTs = data.latest_image_ts;
                    document.getElementById('video_feed').src = 'data:image/jpeg;base64,' + data.latest_image;
                } else if (!data.latest_image && !data.latest_image_ts && !(data.camera && data.camera.streaming)) {
                    lastFrameTs = 0;
                    document.getElementById('video_feed').removeAttribute('src');
                    if (videoCtx && videoOverlay) {
                        videoCtx.clearRect(0, 0, videoOverlay.width, videoOverlay.height);
                    }
                }
                updateCameraUI(data.camera || {});
                document.getElementById('manual_controls').style.display = (data.mode === 'MANUAL') ? 'block' : 'none';
                drawSR(data.vision_debug || {}, data.latest_action, data.mode);
                drawVideoOverlay(data.vision_debug || {});
            }).catch(() => {}).finally(() => { stateBusy = false; });
        }, statePollMs);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD)


@app.route("/debug")
def debug_page():
    debug_path = Path(__file__).resolve().with_name("airpuff_calibration_debug.html")
    if debug_path.is_file():
        return send_file(str(debug_path))
    return (
        "<h3>Debug page not found</h3><p>Missing airpuff_calibration_debug.html beside airpuff_server.py</p>",
        404,
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "models": {
                "llm": CONFIG["LLM_MODEL"],
                "llm_cmd": resolve_llm_model("command"),
                "llm_chat": resolve_llm_model("chat"),
                "vlm": CONFIG["VLM_MODEL"],
                "whisper": CONFIG["WHISPER_MODEL"] if whisper_model else None,
                "yolo": CONFIG["YOLO_MODEL"] if yolo_model else None,
            },
            "config": {
                "enable_llm": CONFIG["ENABLE_LLM"],
                "enable_vlm": CONFIG["ENABLE_VLM"],
                "enable_whisper": CONFIG["ENABLE_WHISPER"] and whisper_model is not None,
                "cmd_fast_path": CONFIG["CMD_FAST_PATH"],
                "cmd_fast_path_max_chars": CONFIG["CMD_FAST_PATH_MAX_CHARS"],
                "num_predict": CONFIG["OLLAMA_NUM_PREDICT"],
                "chat_num_predict": CONFIG["CHAT_NUM_PREDICT"],
                "keyword_fallback": CONFIG["ENABLE_KEYWORD_FALLBACK"],
                "voice_wake_enabled": CONFIG["VOICE_WAKE_ENABLED"],
                "voice_wake_words": CONFIG["VOICE_WAKE_WORDS"],
                "autopilot_idle_sec": CONFIG["AUTOPILOT_IDLE_SEC"],
                "autopilot_mode": CONFIG["AUTOPILOT_MODE"],
                "vision_mode": CONFIG["VISION_MODE"],
                "vision_clahe": CONFIG["VISION_CLAHE"],
                "vision_gamma": CONFIG["VISION_GAMMA"],
                "bright_th": CONFIG["BRIGHT_TH"],
                "bright_frac_th": CONFIG["BRIGHT_FRAC_TH"],
                "bright_scale": CONFIG["BRIGHT_SCALE"],
                "lite_w": CONFIG["LITE_W"],
                "lite_h": CONFIG["LITE_H"],
                "lite_edge_th": CONFIG["LITE_EDGE_TH"],
                "flow_w": CONFIG["FLOW_W"],
                "flow_h": CONFIG["FLOW_H"],
                "flow_mag_th": CONFIG["FLOW_MAG_TH"],
                "yolo_model": CONFIG["YOLO_MODEL"],
                "yolo_imgsz": CONFIG["YOLO_IMGSZ"],
                "yolo_conf": CONFIG["YOLO_CONF"],
                "yolo_obstacle_dist_m": CONFIG["YOLO_OBSTACLE_DIST_M"],
                "yolo_emergency_stop_m": CONFIG["YOLO_EMERGENCY_STOP_M"],
                "stop_condition": CONFIG["STOP_CONDITION"],
                "yolo_detector_ready": vision_state.get("detector_ready"),
                "scan_enabled": CONFIG["VISION_SCAN_ENABLED"],
                "scan_duration_ms": CONFIG["VISION_SCAN_DURATION_MS"],
                "scan_step_ms": CONFIG["VISION_SCAN_STEP_MS"],
                "turn_hold_ms": CONFIG["VISION_TURN_HOLD_MS"],
                "log_path": CONFIG["LOG_PATH"],
                "log_images": CONFIG["LOG_IMAGES"],
            },
        }
    )


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    with metrics_lock:
        return jsonify(metrics)


@app.route("/api/control", methods=["POST"])
def control():
    data = request.json or {}
    with state_lock:
        if "mode" in data:
            state["mode"] = data["mode"]
        if "action" in data:
            state["manual_cmd"] = data["action"]
        state["last_input_ts"] = time.time()
    return jsonify({"status": "ok"})


@app.route("/api/camera", methods=["POST"])
def control_camera():
    data = request.json or {}
    action = str(data.get("action", "") or "").strip().lower()
    if "enabled" in data:
        action = "start" if bool(data.get("enabled")) else "stop"
    result = camera_service_action(action or "status")
    return jsonify(result)


@app.route("/api/camera/status", methods=["GET"])
def get_camera_status():
    refresh_flag = str(request.args.get("refresh", "1")).strip().lower()
    force = refresh_flag not in {"0", "false", "no"}
    return jsonify(refresh_camera_status(force=force))


@app.route("/api/state", methods=["GET"])
def get_state():
    client_image_ts = request.args.get("image_ts", "")
    with state_lock:
        snapshot = dict(state)
    if client_image_ts:
        try:
            known_ts = float(client_image_ts)
            latest_ts = float(snapshot.get("latest_image_ts") or 0.0)
            if latest_ts and abs(latest_ts - known_ts) < 1e-6:
                snapshot["latest_image"] = ""
        except ValueError:
            pass
    existing_debug = snapshot.get("vision_debug")
    suggested_action = ""
    suggested_route = ""
    if isinstance(existing_debug, dict):
        suggested_action = str(existing_debug.get("suggested_action") or "")
        suggested_route = str(existing_debug.get("suggested_route") or "")
    vision_debug = build_vision_debug(vision_action=suggested_action, vision_route=suggested_route)
    latest_image_ts = float(snapshot.get("latest_image_ts") or 0.0)
    detections_image_ts = float(vision_debug.get("detections_image_ts") or 0.0)
    detections_lag_ms: Optional[int] = None
    detections_stale = False
    if latest_image_ts and detections_image_ts:
        lag_sec = latest_image_ts - detections_image_ts
        detections_lag_ms = int(round(lag_sec * 1000.0))
        if lag_sec > float(CONFIG["VISION_DET_SYNC_MAX_LAG_SEC"]) or lag_sec < -0.35:
            detections_stale = True
    elif latest_image_ts and (
        vision_debug.get("detections")
        or vision_debug.get("front_blocked")
        or vision_debug.get("emergency_stop")
        or vision_debug.get("nearest_distance_m") is not None
    ):
        detections_stale = True

    if detections_stale:
        vision_debug["detections"] = []
        vision_debug["nearest_distance_m"] = None
        vision_debug["lane_markings_visible"] = False
        vision_debug["front_blocked"] = False
        vision_debug["emergency_stop"] = False
        vision_debug["emergency_stop_distance_m"] = None
        vision_debug["stop_reason"] = "SYNC_STALE"

    vision_debug["detections_sync_lag_ms"] = detections_lag_ms
    vision_debug["detections_stale"] = detections_stale
    snapshot["vision_debug"] = vision_debug
    snapshot["camera"] = _camera_status_snapshot()
    return jsonify(snapshot)


@app.route("/api/sense", methods=["POST"])
def sense():
    data = request.json or {}
    image_b64 = data.get("image", "") or ""
    audio_b64 = data.get("audio", "") or ""
    text_override = data.get("text", "") or ""
    from_audio = bool(audio_b64 and not text_override)
    image_ts = time.time() if image_b64 else 0.0

    with metrics_lock:
        metrics["requests_total"] += 1
        metrics["seq"] += 1
        seq = metrics["seq"]

    if image_b64:
        with state_lock:
            state["latest_image"] = image_b64
            state["latest_image_ts"] = image_ts

    transcript = ""
    transcript_raw = ""
    wake_info: Dict[str, Any] = {
        "required": bool(CONFIG["VOICE_WAKE_ENABLED"] and from_audio),
        "hit": False,
        "word": "",
        "prompt_only": False,
    }
    asr_ms_val = None
    llm_total_ms = None
    vlm_ms_val = None
    route = "idle"

    with state_lock:
        state["last_error"] = ""

    if text_override:
        transcript = text_override.strip()
    elif audio_b64 and whisper_model and CONFIG["ENABLE_WHISPER"]:
        try:
            transcript, asr_ms = transcribe_audio(audio_b64)
            asr_ms_val = round(asr_ms, 1)
            with metrics_lock:
                metrics["asr_calls"] += 1
            record_latency("asr", asr_ms)
        except Exception as exc:
            with metrics_lock:
                metrics["errors"] += 1
            with state_lock:
                state["last_error"] = f"ASR error: {exc}"

    transcript_raw = transcript
    transcript, wake_info = apply_voice_wake_gate(transcript_raw, from_audio=from_audio)
    wake_consumed_input = bool(transcript_raw) and (
        (not wake_info["required"]) or bool(wake_info["hit"])
    )

    with state_lock:
        state["latest_wake_hit"] = bool(wake_info["hit"])
        state["latest_wake_word"] = str(wake_info["word"] or "")
        state["latest_wake_required"] = bool(wake_info["required"])
        if transcript_raw:
            state["latest_transcript"] = transcript_raw
        if wake_consumed_input:
            state["last_input_ts"] = time.time()

    action = "STOP"
    chat_reply = ""

    with state_lock:
        current_mode = state["mode"]
        manual_cmd = state["manual_cmd"]

    vision_action = ""
    vision_route = ""
    if image_b64:
        vision_action, vision_route, vision_vlm_ms = _run_vision_pipeline(image_b64, frame_ts=image_ts)
        if vision_vlm_ms is not None:
            vlm_ms_val = vision_vlm_ms

    if current_mode == "MANUAL":
        action = manual_cmd
        route = "manual"
        if wake_info["prompt_only"]:
            chat_reply = "我在"
            route = "manual_wake_ack"
        with state_lock:
            if action == "UP":
                state["altitude_setpoint"] += 10
            elif action == "DOWN":
                state["altitude_setpoint"] = max(0, state["altitude_setpoint"] - 10)
    else:
        if wake_info["prompt_only"]:
            action = "STOP"
            chat_reply = "我在"
            route = "wake_ack"
        else:
            fast_cmd = ""
            direct_chat = False
            if transcript and CONFIG["ENABLE_KEYWORD_FALLBACK"]:
                fast_cmd = fast_command_candidate(transcript)
            if transcript and CONFIG["ENABLE_LLM"] and not fast_cmd:
                direct_chat = direct_chat_candidate(transcript)

            if fast_cmd:
                action = fast_cmd
                chat_reply = "[Fast Command Route]"
                route = "cmd_fast"
                with metrics_lock:
                    metrics["fast_cmd_hits"] += 1
            elif transcript and direct_chat and CONFIG["ENABLE_LLM"]:
                try:
                    chat_prompt = build_chat_prompt(transcript)
                    chat_text, chat_ms = ask_chat(chat_prompt)
                    with metrics_lock:
                        metrics["llm_calls"] += 1
                        metrics["llm_chat_calls"] += 1
                    record_latency("llm", chat_ms)
                    llm_total_ms = chat_ms if llm_total_ms is None else llm_total_ms + chat_ms
                    chat_reply = chat_text
                    remember_chat(transcript, chat_reply)
                    route = "chat_direct"
                except Exception as exc:
                    with metrics_lock:
                        metrics["errors"] += 1
                    with state_lock:
                        state["last_error"] = f"Chat error: {exc}"
            elif transcript and CONFIG["ENABLE_LLM"]:
                try:
                    prompt = build_cmd_prompt(transcript)
                    ans, llm_ms = ask_llm(prompt, purpose="command")
                    with metrics_lock:
                        metrics["llm_calls"] += 1
                        metrics["llm_cmd_calls"] += 1
                    record_latency("llm", llm_ms)
                    llm_total_ms = llm_ms if llm_total_ms is None else llm_total_ms + llm_ms
                    parsed = safe_json_extract(ans)
                    if parsed.get("type") == "command":
                        action = str(parsed.get("action", "STOP")).upper()
                        chat_reply = "[Executing Voice Command]"
                        route = "cmd_llm"
                    elif parsed.get("type") == "chat":
                        action = "STOP"
                        route = "chat_llm"
                        try:
                            chat_prompt = build_chat_prompt(transcript)
                            chat_text, chat_ms = ask_chat(chat_prompt)
                            with metrics_lock:
                                metrics["llm_calls"] += 1
                                metrics["llm_chat_calls"] += 1
                            record_latency("llm", chat_ms)
                            llm_total_ms = chat_ms if llm_total_ms is None else llm_total_ms + chat_ms
                            chat_reply = chat_text or parsed.get("reply", "")
                        except Exception:
                            chat_reply = parsed.get("reply", "")
                        remember_chat(transcript, chat_reply)
                    elif CONFIG["ENABLE_KEYWORD_FALLBACK"]:
                        fallback_cmd = keyword_command(transcript)
                        if fallback_cmd:
                            action = fallback_cmd
                            route = "cmd_keyword_fallback"
                            with metrics_lock:
                                metrics["keyword_fallback_hits"] += 1
                except Exception as exc:
                    with metrics_lock:
                        metrics["errors"] += 1
                    with state_lock:
                        state["last_error"] = f"LLM error: {exc}"
                    if CONFIG["ENABLE_KEYWORD_FALLBACK"]:
                        fallback_cmd = keyword_command(transcript)
                        if fallback_cmd:
                            action = fallback_cmd
                            route = "cmd_keyword_fallback"
                            with metrics_lock:
                                metrics["keyword_fallback_hits"] += 1
            elif transcript and CONFIG["ENABLE_KEYWORD_FALLBACK"]:
                fallback_cmd = keyword_command(transcript)
                if fallback_cmd:
                    action = fallback_cmd
                    route = "cmd_keyword_only"
                    with metrics_lock:
                        metrics["keyword_fallback_hits"] += 1
            elif vision_action:
                action = vision_action
                route = vision_route

        # Auto-wander when idle with no input
        with state_lock:
            idle_for = time.time() - state.get("last_input_ts", time.time())
        if (
            action == "STOP"
            and not transcript
            and idle_for >= CONFIG["AUTOPILOT_IDLE_SEC"]
            and not vision_state.get("last_front_blocked")
        ):
            action = autopilot_action(time.time())
            route = "autopilot"

    if vision_state.get("last_emergency_stop"):
        action = "STOP"
        if current_mode == "MANUAL":
            route = "manual_safety_stop"
        elif route.startswith("vision_yolo"):
            route = "vision_yolo_emergency_stop"
        else:
            route = "auto_safety_stop"

    if route == "idle":
        route = "hold"

    vision_debug = build_vision_debug(vision_action=vision_action, vision_route=vision_route)

    with state_lock:
        state["latest_action"] = action
        state["latest_chat"] = chat_reply
        state["vision_debug"] = vision_debug
        state["last_asr_ms"] = asr_ms_val
        state["last_llm_ms"] = round(llm_total_ms, 1) if llm_total_ms is not None else None
        state["last_vlm_ms"] = vlm_ms_val
        state["last_route"] = route
        alt = state["altitude_setpoint"]

    response = {
        "status": "ok",
        "mode": current_mode,
        "action": action,
        "chat": chat_reply,
        "alt_setpoint": alt,
        "asr_ms": asr_ms_val,
        "llm_ms": round(llm_total_ms, 1) if llm_total_ms is not None else None,
        "vlm_ms": vlm_ms_val,
        "route": route,
        "wake": {
            "required": bool(wake_info["required"]),
            "hit": bool(wake_info["hit"]),
            "word": str(wake_info["word"] or ""),
            "prompt_only": bool(wake_info["prompt_only"]),
        },
        "server_ts": time.time(),
        "seq": seq,
    }

    log_entry = {
        "ts": response["server_ts"],
        "seq": seq,
        "mode": state.get("mode"),
        "action": action,
        "route": route,
        "alt_setpoint": alt,
        "asr_ms": response["asr_ms"],
        "llm_ms": response["llm_ms"],
        "vlm_ms": response["vlm_ms"],
        "wake": response["wake"],
        "vision_mode": CONFIG["VISION_MODE"],
        "vision_debug": vision_debug,
    }
    if CONFIG["LOG_IMAGES"] and image_b64:
        log_entry["image"] = image_b64
    if CONFIG["LOG_AUDIO"] and audio_b64:
        log_entry["audio"] = audio_b64
    if CONFIG["LOG_TEXT"] and text_override:
        log_entry["text"] = text_override
    append_log(log_entry)

    return jsonify(response)


if __name__ == "__main__":
    print("------------------------------------------")
    print("AirPuff AI Matrix & Web Dashboard Online!")
    print(f"Access App Control at: http://<Laptop IP>:{CONFIG['SERVER_PORT']}")
    print("------------------------------------------")
    app.run(host=CONFIG["SERVER_HOST"], port=CONFIG["SERVER_PORT"])
