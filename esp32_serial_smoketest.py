import argparse
import sys
import time

try:
    import serial
except Exception as exc:
    print(f"pyserial missing: {exc}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for AirPuff ESP32 serial link.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--action", default="FORWARD")
    parser.add_argument("--alt", type=int, default=0)
    parser.add_argument("--reads", type=int, default=8)
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Give the board a moment in case opening the port triggers reset.
        time.sleep(0.4)

        payload = f"AP,{args.action},{args.alt},{int(time.time() * 1000)}\n"
        print(f"TX {payload.strip()}")
        ser.write(payload.encode("ascii"))
        ser.flush()

        deadline = time.time() + 3.0
        count = 0
        while time.time() < deadline and count < args.reads:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            print(f"RX {line}")
            count += 1


if __name__ == "__main__":
    main()
