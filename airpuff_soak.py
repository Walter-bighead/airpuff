import argparse
import json
import statistics
import time

import requests

try:
    import serial
except Exception:
    serial = None


def append_log(path, entry):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def summarize(values):
    if not values:
        return {"count": 0, "avg_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "avg_ms": round(statistics.mean(values), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def wait_for_line(ser, predicate, timeout_s):
    deadline = time.perf_counter() + timeout_s
    seen = []
    while time.perf_counter() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        recv_ms = int(time.time() * 1000)
        item = {"recv_ms": recv_ms, "line": line}
        seen.append(item)
        if predicate(line):
            return item, seen
    return None, seen


def main():
    parser = argparse.ArgumentParser(description="AirPuff Pi<->Laptop<->ESP32 soak test.")
    parser.add_argument("--server", default="http://10.42.0.1:5000/api/sense")
    parser.add_argument("--serial", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--interval", type=float, default=0.4)
    parser.add_argument("--text", default="前进")
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--failsafe-timeout", type=float, default=1.4)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    if serial is None:
        raise RuntimeError("pyserial not installed")

    health_url = args.server.rsplit("/api/sense", 1)[0] + "/api/health"
    health = requests.get(health_url, timeout=3)
    health.raise_for_status()

    summary = {
        "server": args.server,
        "serial": args.serial,
        "iters": args.iters,
        "health_ok": True,
        "started_at": round(time.time(), 3),
    }

    brain_latencies = []
    ack_latencies = []
    failsafe_latencies = []
    failures = 0

    with serial.Serial(args.serial, args.baud, timeout=0.1) as ser:
        time.sleep(0.4)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        for i in range(1, args.iters + 1):
            entry = {"iter": i, "ts": round(time.time(), 3), "text": args.text}
            try:
                start = time.perf_counter()
                res = requests.post(
                    args.server,
                    json={"text": args.text},
                    timeout=args.request_timeout,
                )
                brain_ms = (time.perf_counter() - start) * 1000.0
                entry["brain_ms"] = round(brain_ms, 2)
                res.raise_for_status()
                data = res.json()
                action = str(data.get("action", "STOP")).upper()
                alt = int(data.get("alt_setpoint", 0))
                route = str(data.get("route", "") or "")
                entry["brain_action"] = action
                entry["alt_setpoint"] = alt
                entry["route"] = route
                brain_latencies.append(brain_ms)
            except Exception as exc:
                failures += 1
                entry["ok"] = False
                entry["error"] = f"brain:{exc}"
                append_log(args.log_path, entry)
                print(f"[{i}/{args.iters}] brain error: {exc}")
                time.sleep(args.interval)
                continue

            ts_ms = int(time.time() * 1000)
            payload = f"AP,{action},{alt},{ts_ms}\n"
            ser.reset_input_buffer()
            ser.write(payload.encode("ascii", errors="ignore"))
            ser.flush()
            entry["serial_tx"] = payload.strip()

            ack_item, ack_seen = wait_for_line(
                ser,
                lambda line: line.startswith(f"ACK,{action},{alt},{ts_ms}"),
                args.ack_timeout,
            )
            entry["serial_rx_pre"] = ack_seen

            if ack_item is None:
                failures += 1
                entry["ok"] = False
                entry["error"] = "serial:ack_timeout"
                append_log(args.log_path, entry)
                print(f"[{i}/{args.iters}] ack timeout")
                time.sleep(args.interval)
                continue

            ack_ms = ack_item["recv_ms"] - ts_ms
            ack_latencies.append(ack_ms)
            entry["ack_ms"] = ack_ms

            fs_item, fs_seen = wait_for_line(
                ser,
                lambda line: line.startswith("EVENT,FAILSAFE,STOP,0"),
                args.failsafe_timeout,
            )
            entry["serial_rx_post"] = fs_seen

            if fs_item is None:
                failures += 1
                entry["ok"] = False
                entry["error"] = "serial:failsafe_timeout"
                append_log(args.log_path, entry)
                print(f"[{i}/{args.iters}] failsafe timeout")
                time.sleep(args.interval)
                continue

            failsafe_ms = fs_item["recv_ms"] - ts_ms
            failsafe_latencies.append(failsafe_ms)
            entry["failsafe_ms"] = failsafe_ms
            entry["ok"] = True
            append_log(args.log_path, entry)
            print(
                f"[{i}/{args.iters}] action={action} route={route or '-'} brain={entry['brain_ms']}ms "
                f"ack={ack_ms}ms failsafe={failsafe_ms}ms"
            )
            time.sleep(args.interval)

    summary["finished_at"] = round(time.time(), 3)
    summary["failures"] = failures
    summary["brain"] = summarize(brain_latencies)
    summary["ack"] = summarize(ack_latencies)
    summary["failsafe"] = summarize(failsafe_latencies)

    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
