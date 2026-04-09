import os
import time
import base64
import json
import tempfile
import threading
from typing import Any, Dict, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional runtime dependency
    WhisperModel = None
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
    "VISION_MODE": os.getenv("AIRPUFF_VISION_MODE", "vlm"),  # vlm | lite | flow | off
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
    "VISION_SCAN_ENABLED": env_bool("AIRPUFF_VISION_SCAN", True),
    "VISION_SCAN_DURATION_MS": env_int("AIRPUFF_SCAN_DURATION_MS", 1200),
    "VISION_SCAN_STEP_MS": env_int("AIRPUFF_SCAN_STEP_MS", 300),
    "VISION_DIFF_TH": env_float("AIRPUFF_VISION_DIFF_TH", 0.03),
    "VISION_TURN_HOLD_MS": env_int("AIRPUFF_TURN_HOLD_MS", 600),
    "OLLAMA_NUM_PREDICT": env_int("AIRPUFF_NUM_PREDICT", 16),
    "ENABLE_KEYWORD_FALLBACK": env_bool("AIRPUFF_KEYWORD_FALLBACK", True),
    "AUTOPILOT_IDLE_SEC": env_int("AIRPUFF_AUTOPILOT_IDLE_SEC", 5),
    "AUTOPILOT_MODE": os.getenv("AIRPUFF_AUTOPILOT_MODE", "wander"),  # wander | circle | off
    "AUTOPILOT_STEP_SEC": env_int("AIRPUFF_AUTOPILOT_STEP_SEC", 2),
    "AUTOPILOT_FORWARD_STEPS": env_int("AIRPUFF_AUTOPILOT_FORWARD_STEPS", 5),
    "AUTOPILOT_TURN_ACTION": os.getenv("AIRPUFF_AUTOPILOT_TURN_ACTION", "RIGHT"),
    "CHAT_HISTORY_MAX": env_int("AIRPUFF_CHAT_HISTORY_MAX", 6),
    "CHAT_NUM_PREDICT": env_int("AIRPUFF_CHAT_NUM_PREDICT", 48),
    "LOG_PATH": os.getenv("AIRPUFF_LOG_PATH", ""),
    "LOG_IMAGES": env_bool("AIRPUFF_LOG_IMAGES", False),
    "LOG_AUDIO": env_bool("AIRPUFF_LOG_AUDIO", False),
    "LOG_TEXT": env_bool("AIRPUFF_LOG_TEXT", True),
    "SERVER_HOST": os.getenv("AIRPUFF_HOST", "0.0.0.0"),
    "SERVER_PORT": env_int("AIRPUFF_PORT", 5000),
}


state_lock = threading.Lock()
state: Dict[str, Any] = {
    "mode": "AUTO",  # "AUTO" or "MANUAL"
    "manual_cmd": "STOP",
    "latest_image": "",
    "latest_transcript": "",
    "latest_chat": "",
    "latest_action": "STOP",
    "altitude_setpoint": 0,
    "last_asr_ms": None,
    "last_llm_ms": None,
    "last_vlm_ms": None,
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
    "last_vision_ts": 0.0,
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


def _decide_turn(center_val: float, left_val: float, right_val: float, obstacle_th: float) -> str:
    now = time.time()
    obstacle = center_val >= obstacle_th
    if not obstacle:
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


HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>AirPuff Motherbase</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { font-family: -apple-system, sans-serif; text-align: center; background: #111; color: #eee; margin: 0; padding: 10px; }
        h2 { color: #00f0ff; }
        .btn { padding: 20px; margin: 5px; font-size: 16px; font-weight: bold; border-radius: 12px; border: none; cursor: pointer; color: white; touch-action: manipulation; }
        .btn-green { background: #00b894; }
        .btn-red { background: #d63031; }
        .btn-blue { background: #0984e3; }
        img { width: 100%; max-width: 520px; border-radius: 8px; border: 2px solid #555; }
        canvas { width: 100%; max-width: 520px; border-radius: 10px; border: 2px solid #333; background: #0b0f14; }
        .panel { background: #222; padding: 15px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; max-width: 300px; margin: 0 auto; }
        .log { text-align: left; font-family: monospace; font-size: 14px; color: #00f0ff; background: #000; padding: 10px; border-radius: 5px; height: 120px; overflow-y: auto; }
        .kv { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; text-align: left; font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <h2>AirPuff Terminal</h2>

    <div class="panel">
        <h3 style="margin:5px 0;">Pilot: <span id="mode_label" style="color:#00b894;">AI AUTO</span></h3>
        <button class="btn btn-green" onclick="setMode('AUTO')">AI Mode</button>
        <button class="btn btn-red" onclick="setMode('MANUAL')">Manual Mode</button>
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
        <canvas id="sr_canvas" width="480" height="300"></canvas>
    </div>

    <div class="panel">
        <img id="video_feed" src="" alt="Awaiting Video Stream..."/>
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
        function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
        function lerp(a, b, t) { return a + (b - a) * t; }
        function drawSR(vd, action, mode) {
            if (!srCtx || !srCanvas) return;
            const ctx = srCtx;
            const w = srCanvas.width;
            const h = srCanvas.height;
            ctx.clearRect(0, 0, w, h);
            const sky = ctx.createLinearGradient(0, 0, 0, h);
            sky.addColorStop(0, '#081018');
            sky.addColorStop(0.55, '#060b12');
            sky.addColorStop(1, '#030407');
            ctx.fillStyle = sky;
            ctx.fillRect(0, 0, w, h);

            ctx.fillStyle = 'rgba(0,120,255,0.08)';
            ctx.beginPath();
            ctx.ellipse(w / 2, h * 0.22, w * 0.46, h * 0.18, 0, 0, Math.PI * 2);
            ctx.fill();

            const roadTopY = h * 0.18;
            const roadBotY = h * 0.96;
            const roadTopW = w * 0.38;
            const roadBotW = w * 0.95;
            const cx = w / 2;

            ctx.fillStyle = '#0f151d';
            ctx.beginPath();
            ctx.moveTo(cx - roadTopW / 2, roadTopY);
            ctx.lineTo(cx + roadTopW / 2, roadTopY);
            ctx.lineTo(cx + roadBotW / 2, roadBotY);
            ctx.lineTo(cx - roadBotW / 2, roadBotY);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = 'rgba(255,120,80,0.28)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx - roadTopW / 2, roadTopY);
            ctx.lineTo(cx - roadBotW / 2, roadBotY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx + roadTopW / 2, roadTopY);
            ctx.lineTo(cx + roadBotW / 2, roadBotY);
            ctx.stroke();

            ctx.lineWidth = 2;
            ctx.setLineDash([6, 10]);
            ctx.strokeStyle = 'rgba(255,255,255,0.25)';
            for (let i = 1; i <= 2; i++) {
                const t = i / 3;
                const xTop = lerp(cx - roadTopW / 2, cx + roadTopW / 2, t);
                const xBot = lerp(cx - roadBotW / 2, cx + roadBotW / 2, t);
                ctx.beginPath();
                ctx.moveTo(xTop, roadTopY + 6);
                ctx.lineTo(xBot, roadBotY - 6);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            ctx.strokeStyle = 'rgba(0,180,255,0.08)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 4; i++) {
                const y = lerp(roadTopY + 6, roadBotY - 6, i / 5);
                const halfW = lerp(roadTopW, roadBotW, i / 5) / 2;
                ctx.beginPath();
                ctx.moveTo(cx - halfW, y);
                ctx.lineTo(cx + halfW, y);
                ctx.stroke();
            }

            if (mode === 'AUTO') {
                let targetOffset = 0;
                if (action === 'LEFT') targetOffset = -0.22;
                if (action === 'RIGHT') targetOffset = 0.22;
                const topX = cx + roadTopW * targetOffset;
                const botX = cx + roadBotW * targetOffset;
                const pathWTop = roadTopW * 0.08;
                const pathWBot = roadBotW * 0.16;
                ctx.fillStyle = 'rgba(20,140,255,0.45)';
                ctx.beginPath();
                ctx.moveTo(botX - pathWBot / 2, roadBotY);
                ctx.quadraticCurveTo(cx, h * 0.58, topX - pathWTop / 2, roadTopY + 10);
                ctx.lineTo(topX + pathWTop / 2, roadTopY + 10);
                ctx.quadraticCurveTo(cx, h * 0.58, botX + pathWBot / 2, roadBotY);
                ctx.closePath();
                ctx.fill();
                ctx.strokeStyle = 'rgba(90,190,255,0.9)';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            function laneXAtY(lane, y) {
                const t = clamp((y - roadTopY) / (roadBotY - roadTopY), 0, 1);
                const halfW = lerp(roadTopW, roadBotW, t) / 2;
                const laneW = (halfW * 2) / 3;
                return cx - halfW + laneW * (lane + 0.5);
            }

            function drawShadow(x, y, rx, ry, alpha) {
                ctx.fillStyle = `rgba(0,0,0,${alpha})`;
                ctx.beginPath();
                ctx.ellipse(x, y + 6, rx, ry, 0, 0, Math.PI * 2);
                ctx.fill();
            }

            function drawTree(x, y, s) {
                drawShadow(x, y, 12 * s, 5 * s, 0.35);
                ctx.fillStyle = '#3a5f2f';
                ctx.beginPath();
                ctx.arc(x, y - 10 * s, 12 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#2d4a24';
                ctx.beginPath();
                ctx.arc(x + 6 * s, y - 14 * s, 8 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#6b4a2b';
                ctx.fillRect(x - 2 * s, y - 2 * s, 4 * s, 10 * s);
            }

            function drawPerson(x, y, s) {
                drawShadow(x, y, 9 * s, 4 * s, 0.32);
                ctx.fillStyle = '#dfe6e9';
                ctx.beginPath();
                ctx.arc(x, y - 10 * s, 4 * s, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#74b9ff';
                ctx.fillRect(x - 3 * s, y - 6 * s, 6 * s, 12 * s);
                ctx.fillStyle = '#b2bec3';
                ctx.fillRect(x - 6 * s, y + 6 * s, 4 * s, 8 * s);
                ctx.fillRect(x + 2 * s, y + 6 * s, 4 * s, 8 * s);
            }

            function drawSeat(x, y, s) {
                drawShadow(x, y, 12 * s, 5 * s, 0.34);
                ctx.fillStyle = '#7f8c8d';
                ctx.fillRect(x - 8 * s, y - 6 * s, 16 * s, 8 * s);
                ctx.fillStyle = '#95a5a6';
                ctx.fillRect(x - 8 * s, y - 16 * s, 16 * s, 10 * s);
                ctx.fillStyle = '#2d3436';
                ctx.fillRect(x - 6 * s, y + 2 * s, 12 * s, 3 * s);
            }

            const th = (vd && typeof vd.obstacle_th === 'number' && vd.obstacle_th > 0) ? vd.obstacle_th : 1;
            const leftVal = (vd && typeof vd.left_val === 'number') ? vd.left_val : 0;
            const centerVal = (vd && typeof vd.center_val === 'number') ? vd.center_val : 0;
            const rightVal = (vd && typeof vd.right_val === 'number') ? vd.right_val : 0;

            function drawActor(lane, val, kind) {
                const intensity = clamp(val / th, 0, 1.3);
                if (intensity < 0.05) return;
                const depth = clamp(intensity, 0, 1);
                const y = lerp(roadTopY + 20, roadBotY - 55, depth);
                const scale = 0.45 + depth * 0.9;
                const x = laneXAtY(lane, y);
                if (kind === 'tree') drawTree(x, y, scale);
                if (kind === 'person') drawPerson(x, y, scale);
                if (kind === 'seat') drawSeat(x, y, scale);
            }

            drawActor(0, leftVal, 'tree');
            drawActor(1, centerVal, 'person');
            drawActor(2, rightVal, 'seat');

            const selfY = roadBotY - 18;
            const carW = roadBotW * 0.12;
            const carH = carW * 0.55;
            drawShadow(cx, selfY + 6, carW * 0.45, carH * 0.25, 0.5);
            ctx.fillStyle = '#dfe6e9';
            ctx.fillRect(cx - carW / 2, selfY - carH, carW, carH);
            ctx.fillStyle = '#b2bec3';
            ctx.fillRect(cx - carW / 3, selfY - carH * 0.8, carW * 0.66, carH * 0.5);
            ctx.fillStyle = 'rgba(0,180,255,0.8)';
            ctx.fillRect(cx - carW / 2, selfY - 2, 6, 3);
            ctx.fillRect(cx + carW / 2 - 6, selfY - 2, 6, 3);

            if (vd && vd.scan_active) {
                ctx.fillStyle = 'rgba(0,240,255,0.85)';
                ctx.font = '12px monospace';
                const dir = vd.scan_dir || '';
                ctx.fillText(`SCAN ${dir}`, 10, 18);
            }

            if (action) {
                ctx.fillStyle = 'rgba(0,180,255,0.8)';
                ctx.font = '12px monospace';
                ctx.fillText(`CMD ${action}`, w - 100, 18);
            }
        }
        let stateBusy = false;
        setInterval(() => {
            if (stateBusy) return;
            stateBusy = true;
            fetch('/api/state?ts=' + Date.now(), { cache: 'no-store' }).then(r => r.json()).then(data => {
                document.getElementById('mode_label').innerText = data.mode;
                document.getElementById('mode_label').style.color = data.mode === 'AUTO' ? '#00b894' : '#d63031';
                document.getElementById('action').innerText = data.latest_action;
                document.getElementById('heard').innerText = data.latest_transcript;
                document.getElementById('chat').innerText = data.latest_chat;
                document.getElementById('asr_ms').innerText = data.last_asr_ms ?? '-';
                document.getElementById('llm_ms').innerText = data.last_llm_ms ?? '-';
                document.getElementById('vlm_ms').innerText = data.last_vlm_ms ?? '-';
                if (data.latest_image) {
                    document.getElementById('video_feed').src = 'data:image/jpeg;base64,' + data.latest_image;
                }
                document.getElementById('manual_controls').style.display = (data.mode === 'MANUAL') ? 'block' : 'none';
                drawSR(data.vision_debug || {}, data.latest_action, data.mode);
            }).catch(() => {}).finally(() => { stateBusy = false; });
        }, 100);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD)


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


@app.route("/api/state", methods=["GET"])
def get_state():
    with state_lock:
        return jsonify(state)


@app.route("/api/sense", methods=["POST"])
def sense():
    data = request.json or {}
    image_b64 = data.get("image", "") or ""
    audio_b64 = data.get("audio", "") or ""
    text_override = data.get("text", "") or ""

    with metrics_lock:
        metrics["requests_total"] += 1
        metrics["seq"] += 1
        seq = metrics["seq"]

    if image_b64:
        with state_lock:
            state["latest_image"] = image_b64

    transcript = ""
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

    if transcript:
        with state_lock:
            state["latest_transcript"] = transcript
            state["last_input_ts"] = time.time()

    action = "STOP"
    chat_reply = ""

    with state_lock:
        current_mode = state["mode"]
        manual_cmd = state["manual_cmd"]

    if current_mode == "MANUAL":
        action = manual_cmd
        route = "manual"
        with state_lock:
            if action == "UP":
                state["altitude_setpoint"] += 10
            elif action == "DOWN":
                state["altitude_setpoint"] = max(0, state["altitude_setpoint"] - 10)
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
        elif image_b64 and CONFIG["VISION_MODE"] == "lite":
            now = time.time()
            if (now - vision_state.get("last_lite_ts", 0.0)) * 1000.0 >= 200:
                vision_state["last_lite_ts"] = now
                lite_cmd = lite_vision_action(image_b64)
                if lite_cmd:
                    action = lite_cmd
                    route = "vision_lite"
        elif image_b64 and CONFIG["VISION_MODE"] == "flow":
            now = time.time()
            if (now - vision_state.get("last_flow_ts", 0.0)) * 1000.0 >= CONFIG["FLOW_MIN_INTERVAL_MS"]:
                vision_state["last_flow_ts"] = now
                flow_cmd = flow_vision_action(image_b64)
                if flow_cmd:
                    action = flow_cmd
                    route = "vision_flow"
        elif image_b64 and CONFIG["ENABLE_VLM"] and CONFIG["VISION_MODE"] == "vlm":
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
                    vlm_ms_val = round(vlm_ms, 1)
                    upper = ans.upper()
                    for cmd in ["FORWARD", "LEFT", "RIGHT", "STOP", "BACKWARD"]:
                        if cmd in upper:
                            action = cmd
                            route = "vision_vlm"
                            break
                except Exception as exc:
                    with metrics_lock:
                        metrics["errors"] += 1
                    with state_lock:
                        state["last_error"] = f"VLM error: {exc}"

        # Auto-wander when idle with no input
        with state_lock:
            idle_for = time.time() - state.get("last_input_ts", time.time())
        if action == "STOP" and not transcript and idle_for >= CONFIG["AUTOPILOT_IDLE_SEC"]:
            action = autopilot_action(time.time())
            route = "autopilot"

    if route == "idle":
        route = "hold"

    vision_debug = {
        "mode": vision_state.get("last_vision_mode"),
        "bright_frac": vision_state.get("last_bright_frac"),
        "obstacle_th_base": vision_state.get("last_obstacle_th_base"),
        "obstacle_th": vision_state.get("last_obstacle_th"),
        "center_val": vision_state.get("last_center_val"),
        "left_val": vision_state.get("last_left_val"),
        "right_val": vision_state.get("last_right_val"),
        "vision_ts": vision_state.get("last_vision_ts"),
        "scan_active": vision_state.get("scan_active"),
        "scan_dir": vision_state.get("scan_dir"),
    }

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
        "action": action,
        "chat": chat_reply,
        "alt_setpoint": alt,
        "asr_ms": asr_ms_val,
        "llm_ms": round(llm_total_ms, 1) if llm_total_ms is not None else None,
        "vlm_ms": vlm_ms_val,
        "route": route,
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
