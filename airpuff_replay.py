import argparse
import json
import time
import requests


def main():
    parser = argparse.ArgumentParser(description="Replay logged AirPuff sensory data.")
    parser.add_argument("--log", required=True, help="Path to JSONL log file.")
    parser.add_argument("--url", default="http://127.0.0.1:5000/api/sense")
    parser.add_argument("--interval", type=float, default=0.2, help="Fixed replay interval (seconds).")
    parser.add_argument("--realtime", action="store_true", help="Use timestamps in log for timing.")
    args = parser.parse_args()

    last_ts = None
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            payload = {}
            if "image" in entry:
                payload["image"] = entry["image"]
            if "audio" in entry:
                payload["audio"] = entry["audio"]
            if "text" in entry:
                payload["text"] = entry["text"]

            try:
                requests.post(args.url, json=payload, timeout=30)
            except Exception:
                pass

            if args.realtime and "ts" in entry:
                if last_ts is not None:
                    delay = max(0.0, float(entry["ts"]) - last_ts)
                    time.sleep(delay)
                last_ts = float(entry["ts"])
            else:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
