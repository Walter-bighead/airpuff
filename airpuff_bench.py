import argparse
import base64
import io
import json
import os
import platform
import statistics
import tempfile
import time
import wave

import requests

try:
    import cv2
except Exception:
    cv2 = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


def make_dummy_image_b64():
    if cv2 is None:
        return ""
    img = (255 * (cv2.getGaussianKernel(240, 60) @ cv2.getGaussianKernel(320, 80).T)).astype("uint8")
    img = cv2.merge([img, img, img])
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def make_silence_wav(seconds=3, rate=16000):
    frames = b"\x00\x00" * int(rate * seconds)
    wav_io = io.BytesIO()
    wf = wave.open(wav_io, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(frames)
    wf.close()
    return wav_io.getvalue()


def bench_ollama(api_url, model, prompt, image_b64=None, iters=3, timeout=120, num_predict=None):
    latencies = []
    errors = 0
    for _ in range(iters):
        payload = {"model": model, "prompt": prompt, "stream": False}
        if num_predict is not None:
            payload["options"] = {"num_predict": int(num_predict)}
        if image_b64:
            payload["images"] = [image_b64]
        start = time.perf_counter()
        try:
            requests.post(api_url, json=payload, timeout=timeout).json()
            latencies.append((time.perf_counter() - start) * 1000.0)
        except Exception:
            errors += 1
    return latencies, errors


def bench_whisper(model_name, device, compute_type, iters=1):
    if WhisperModel is None:
        return [], 1
    latencies = []
    errors = 0
    wav_bytes = make_silence_wav()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        return [], iters
    for _ in range(iters):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_audio = f.name
        start = time.perf_counter()
        try:
            list(model.transcribe(temp_audio))
            latencies.append((time.perf_counter() - start) * 1000.0)
        except Exception:
            errors += 1
        finally:
            try:
                os.remove(temp_audio)
            except OSError:
                pass
    return latencies, errors


def summarize(name, latencies, errors):
    result = {
        "name": name,
        "count": len(latencies),
        "errors": errors,
        "avg_ms": None,
        "p95_ms": None,
        "min_ms": None,
        "max_ms": None,
    }
    if latencies:
        result["avg_ms"] = round(statistics.mean(latencies), 2)
        result["min_ms"] = round(min(latencies), 2)
        result["max_ms"] = round(max(latencies), 2)
        if len(latencies) >= 2:
            result["p95_ms"] = round(statistics.quantiles(latencies, n=20)[-1], 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="AirPuff laptop performance benchmark.")
    parser.add_argument("--ollama", default=os.getenv("AIRPUFF_OLLAMA_API", "http://127.0.0.1:11434/api/generate"))
    parser.add_argument("--llm", default=os.getenv("AIRPUFF_LLM_MODEL", "lfm2:24b"))
    parser.add_argument("--vlm", default=os.getenv("AIRPUFF_VLM_MODEL", "minicpm-v"))
    parser.add_argument("--whisper", default=os.getenv("AIRPUFF_WHISPER_MODEL", "tiny"))
    parser.add_argument("--whisper-device", default=os.getenv("AIRPUFF_WHISPER_DEVICE", "cpu"))
    parser.add_argument("--whisper-compute", default=os.getenv("AIRPUFF_WHISPER_COMPUTE", "int8"))
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--num-predict", type=int, default=int(os.getenv("AIRPUFF_NUM_PREDICT", "32")))
    parser.add_argument("--asr-iters", type=int, default=1)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    report = {
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "results": [],
    }

    if not args.skip_llm:
        llm_lat, llm_err = bench_ollama(
            args.ollama,
            args.llm,
            "Return JSON {\"type\":\"command\",\"action\":\"FORWARD\"}.",
            iters=args.iters,
            num_predict=args.num_predict,
        )
        report["results"].append(summarize("llm_command", llm_lat, llm_err))

    if not args.skip_vlm:
        image_b64 = make_dummy_image_b64()
        vlm_lat, vlm_err = bench_ollama(
            args.ollama,
            args.vlm,
            "Answer only one word: FORWARD, LEFT, RIGHT, STOP.",
            image_b64=image_b64,
            iters=args.iters,
            timeout=180,
            num_predict=args.num_predict,
        )
        report["results"].append(summarize("vlm_obstacle", vlm_lat, vlm_err))

    if not args.skip_asr:
        asr_lat, asr_err = bench_whisper(
            args.whisper,
            device=args.whisper_device,
            compute_type=args.whisper_compute,
            iters=args.asr_iters,
        )
        report["results"].append(summarize("asr_whisper", asr_lat, asr_err))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
