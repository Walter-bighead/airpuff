import argparse
import statistics
import time

import requests

try:
    import serial
except Exception:
    serial = None


def check_laptop(server_url, iters):
    latencies = []
    health_url = server_url.rsplit("/api/sense", 1)[0] + "/api/health"
    health = requests.get(health_url, timeout=3)
    health.raise_for_status()
    for _ in range(iters):
        start = time.perf_counter()
        res = requests.post(server_url, json={"text": "前进"}, timeout=10)
        res.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    return {
        "health_ok": True,
        "avg_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def check_esp32(port, baud):
    if serial is None:
        raise RuntimeError("pyserial not installed")
    with serial.Serial(port, baud, timeout=0.6) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.4)
        payload = f"AP,FORWARD,0,{int(time.time() * 1000)}\n"
        ser.write(payload.encode("ascii"))
        ser.flush()
        lines = []
        deadline = time.time() + 2.0
        while time.time() < deadline and len(lines) < 6:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
        return {"tx": payload.strip(), "rx": lines}


def main():
    parser = argparse.ArgumentParser(description="AirPuff system debug on Pi.")
    parser.add_argument("--server", default="http://192.168.31.240:5000/api/sense")
    parser.add_argument("--serial", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    print("== Laptop Health / Sense ==")
    try:
        print(check_laptop(args.server, args.iters))
    except Exception as exc:
        print({"health_ok": False, "error": str(exc)})

    print("== ESP32 Serial ==")
    try:
        print(check_esp32(args.serial, args.baud))
    except Exception as exc:
        print({"serial_ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
