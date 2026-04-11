import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time
import wave

import requests

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    import serial
except Exception:
    serial = None

SERVER_URL = os.getenv("AIRPUFF_SERVER_URL", "http://10.42.0.1:5000/api/sense")
VALID_ACTIONS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "UP", "DOWN"}


class FailsafeController:
    def __init__(self, link_timeout_s, failsafe_action):
        self.link_timeout_s = link_timeout_s
        self.failsafe_action = failsafe_action
        self.last_rx_ts = 0.0
        self.last_action = "STOP"

    def update_rx(self, action, ts):
        if action in VALID_ACTIONS:
            self.last_action = action
            self.last_rx_ts = ts

    def effective_action(self, now):
        if self.last_rx_ts <= 0 or (now - self.last_rx_ts) > self.link_timeout_s:
            return self.failsafe_action
        return self.last_action


class SerialMonitor:
    def __init__(self, serial_dev):
        self.serial_dev = serial_dev
        self.stop_event = threading.Event()
        self.lines = queue.Queue(maxsize=32)
        self.thread = None

    def start(self):
        if self.serial_dev is None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _push_line(self, line):
        if not line:
            return
        try:
            self.lines.put_nowait(line)
        except queue.Full:
            try:
                self.lines.get_nowait()
            except queue.Empty:
                pass
            try:
                self.lines.put_nowait(line)
            except queue.Full:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                raw = self.serial_dev.readline()
            except Exception:
                time.sleep(0.05)
                continue
            if not raw:
                time.sleep(0.02)
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            self._push_line(line)

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return items


def normalize_failsafe(value):
    value = (value or "STOP").upper().strip()
    if value in {"HOVER", "STOP"}:
        return "STOP"
    if value in {"DESCEND", "DOWN"}:
        return "DOWN"
    return "STOP"


def init_camera():
    if cv2 is None:
        return None
    try:
        return cv2.VideoCapture(0)
    except Exception:
        return None


def init_audio():
    if pyaudio is None:
        return None
    try:
        return pyaudio.PyAudio()
    except Exception:
        return None


def init_serial(port, baud):
    if not port:
        return None
    if serial is None:
        print("[!] pyserial not installed, serial output disabled.")
        return None
    try:
        return serial.Serial(port, baud, timeout=0.1)
    except Exception as exc:
        print(f"[!] Serial open failed: {exc}")
        return None

def capture_image(cap, width, height, quality):
    if cap is None or cv2 is None or not cap.isOpened():
        return ""
    ret, frame = cap.read()
    if not ret:
        return ""
    frame = cv2.resize(frame, (width, height))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")

def capture_audio(p, record_seconds=3):
    if p is None or pyaudio is None:
        return ""
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        frames = []
        for _ in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
    except Exception:
        return ""

    wav_io = io.BytesIO()
    wf = wave.open(wav_io, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    return base64.b64encode(wav_io.getvalue()).decode("utf-8")


def encode_image_file(path):
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def tts_speak(text, enable_tts):
    if text and enable_tts:
        print(f"\n[AI SPEAKS]: {text}")
        # Note: requires 'sudo apt install espeak' on Pi
        os.system(f'espeak -v en "{text}" 2>/dev/null &')


def append_log(path, entry):
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception:
        pass


def apply_action(action, alt_setpoint, serial_dev=None, dry_run=False):
    # Placeholder for future hardware output (ESC/ESP32).
    if dry_run:
        return
    if serial_dev is None:
        return
    try:
        ts_ms = int(time.time() * 1000)
        payload = f"AP,{action},{int(alt_setpoint)},{ts_ms}\\n"
        serial_dev.write(payload.encode("ascii", errors="ignore"))
    except Exception:
        pass


def main_loop(mode, cap, audio, serial_dev, serial_monitor, args):
    print("--- AirPuff Sensory & Execution Node Online ---")
    text_pending = args.text.strip() if args.text else ""
    image_b64_cached = encode_image_file(args.image) if args.image else ""
    text_once = args.text_mode == "once"

    failsafe = FailsafeController(args.link_timeout, normalize_failsafe(args.failsafe_action))
    loop_idx = 0
    while True:
        loop_idx += 1
        log_entry = {
            "loop": loop_idx,
            "ts": round(time.time(), 3),
            "mode": mode,
        }
        img_b64 = ""
        aud_b64 = ""
        text = ""

        if mode == "simulate":
            img_b64 = image_b64_cached
            if text_pending:
                text = text_pending
                if text_once:
                    text_pending = ""
        else:
            img_data = {"b64": ""}
            aud_data = {"b64": ""}
            if not args.disable_video:
                t1 = threading.Thread(
                    target=lambda: img_data.update(
                        {"b64": capture_image(cap, args.width, args.height, args.jpg_quality)}
                    )
                )
            else:
                t1 = None
            if not args.disable_audio:
                t2 = threading.Thread(target=lambda: aud_data.update({"b64": capture_audio(audio, args.audio_seconds)}))
            else:
                t2 = None
            if t1:
                t1.start()
            if t2:
                t2.start()
            if t1:
                t1.join()
            if t2:
                t2.join()
            img_b64 = img_data["b64"]
            aud_b64 = aud_data["b64"]
            if text_pending:
                text = text_pending
                if text_once:
                    text_pending = ""

        payload = {"image": img_b64, "audio": aud_b64, "text": text}
        log_entry["payload"] = {
            "has_image": bool(img_b64),
            "has_audio": bool(aud_b64),
            "text": text,
        }

        try:
            start = time.time()
            res = requests.post(SERVER_URL, json=payload, timeout=args.request_timeout)
            brain_ms = round((time.time() - start) * 1000.0, 2)
            if res.status_code == 200:
                data = res.json()
                action = data.get("action", "STOP")
                chat_msg = data.get("chat", "")
                alt_setpoint = data.get("alt_setpoint", 0)

                print(f"[Brain in {brain_ms/1000.0:.2f}s] CMD: {action} | ALT: {alt_setpoint}")

                if chat_msg:
                    tts_speak(chat_msg, args.enable_tts)

                now = time.time()
                failsafe.update_rx(action, now)
                effective = failsafe.effective_action(now)
                apply_action(effective, alt_setpoint, serial_dev=serial_dev, dry_run=args.dry_run)
                serial_lines = serial_monitor.drain() if serial_monitor else []
                for line in serial_lines:
                    print(f"[ESP32] {line}")
                log_entry.update(
                    {
                        "ok": True,
                        "brain_ms": brain_ms,
                        "brain_action": action,
                        "effective_action": effective,
                        "alt_setpoint": alt_setpoint,
                        "chat": chat_msg,
                        "serial_lines": serial_lines,
                    }
                )
            else:
                log_entry.update(
                    {
                        "ok": False,
                        "brain_ms": brain_ms,
                        "error": f"http_{res.status_code}",
                    }
                )
        except Exception:
            print("[-] Brain connection lost...")
            now = time.time()
            effective = failsafe.effective_action(now)
            apply_action(effective, 0, serial_dev=serial_dev, dry_run=args.dry_run)
            serial_lines = serial_monitor.drain() if serial_monitor else []
            for line in serial_lines:
                print(f"[ESP32] {line}")
            log_entry.update(
                {
                    "ok": False,
                    "effective_action": effective,
                    "serial_lines": serial_lines,
                    "error": "brain_connection_lost",
                }
            )

        append_log(args.log_path, log_entry)
        if args.max_loops > 0 and loop_idx >= args.max_loops:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="AirPuff client (hardware or simulate).")
        parser.add_argument("--mode", choices=["hardware", "simulate"], default=os.getenv("AIRPUFF_MODE", "hardware"))
        parser.add_argument("--text", default=os.getenv("AIRPUFF_TEXT", ""))
        parser.add_argument(
            "--text-mode",
            choices=["once", "repeat"],
            default=os.getenv("AIRPUFF_TEXT_MODE", "once"),
        )
        parser.add_argument("--image", default=os.getenv("AIRPUFF_IMAGE", ""))
        parser.add_argument("--interval", type=float, default=float(os.getenv("AIRPUFF_INTERVAL", "0.5")))
        parser.add_argument("--audio-seconds", type=int, default=int(os.getenv("AIRPUFF_AUDIO_SECONDS", "3")))
        parser.add_argument("--disable-audio", action="store_true", default=os.getenv("AIRPUFF_DISABLE_AUDIO", "0") == "1")
        parser.add_argument("--disable-video", action="store_true", default=os.getenv("AIRPUFF_DISABLE_VIDEO", "0") == "1")
        parser.add_argument("--width", type=int, default=int(os.getenv("AIRPUFF_WIDTH", "320")))
        parser.add_argument("--height", type=int, default=int(os.getenv("AIRPUFF_HEIGHT", "240")))
        parser.add_argument("--jpg-quality", type=int, default=int(os.getenv("AIRPUFF_JPG_QUALITY", "60")))
        parser.add_argument("--enable-tts", action="store_true", default=os.getenv("AIRPUFF_TTS", "0") == "1")
        parser.add_argument("--link-timeout", type=float, default=float(os.getenv("AIRPUFF_LINK_TIMEOUT", "2.0")))
        parser.add_argument("--failsafe-action", default=os.getenv("AIRPUFF_FAILSAFE", "STOP"))
        parser.add_argument("--dry-run", action="store_true", default=os.getenv("AIRPUFF_DRY_RUN", "1") == "1")
        parser.add_argument("--serial", default=os.getenv("AIRPUFF_SERIAL", ""))
        parser.add_argument("--baud", type=int, default=int(os.getenv("AIRPUFF_BAUD", "115200")))
        parser.add_argument("--request-timeout", type=float, default=float(os.getenv("AIRPUFF_REQUEST_TIMEOUT", "60")))
        parser.add_argument("--max-loops", type=int, default=int(os.getenv("AIRPUFF_MAX_LOOPS", "0")))
        parser.add_argument("--log-path", default=os.getenv("AIRPUFF_CLIENT_LOG", ""))
        args = parser.parse_args()

        cap = init_camera() if args.mode == "hardware" and not args.disable_video else None
        audio = init_audio() if args.mode == "hardware" and not args.disable_audio else None
        serial_dev = init_serial(args.serial, args.baud)
        serial_monitor = SerialMonitor(serial_dev)
        serial_monitor.start()
        main_loop(args.mode, cap, audio, serial_dev, serial_monitor, args)
    except KeyboardInterrupt:
        if "cap" in globals() and cap:
            cap.release()
        if "audio" in globals() and audio:
            audio.terminate()
        if "serial_monitor" in globals() and serial_monitor:
            serial_monitor.stop()
        if "serial_dev" in globals() and serial_dev:
            serial_dev.close()
        sys.exit(0)
