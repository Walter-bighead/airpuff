import argparse
import base64
import os
import time

try:
    import serial
except Exception as exc:
    raise SystemExit(f"pyserial missing: {exc}")


def chunk_string(text, size):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def read_until_quiet(ser, quiet_s=0.15, timeout_s=2.0):
    end = time.time() + timeout_s
    last_data = time.time()
    chunks = []
    while time.time() < end:
        waiting = ser.in_waiting or 1
        data = ser.read(waiting)
        if data:
            chunks.append(data)
            last_data = time.time()
        elif time.time() - last_data >= quiet_s:
            break
    return b"".join(chunks)


def write_text(ser, text):
    ser.write(text.encode("utf-8"))
    ser.flush()
    time.sleep(0.01)


def enter_paste_mode(ser):
    ser.write(b"\r\x03\x03\x02")
    ser.flush()
    time.sleep(0.25)
    read_until_quiet(ser, timeout_s=0.6)
    ser.write(b"\x05")
    ser.flush()
    time.sleep(0.15)
    read_until_quiet(ser, timeout_s=0.6)


def upload_file(ser, local_path, remote_name):
    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    enter_paste_mode(ser)
    write_text(ser, "import ubinascii\r\n")
    write_text(ser, "f=open({!r},'wb')\r\n".format(remote_name))
    for chunk in chunk_string(encoded, 24):
        write_text(ser, "f.write(ubinascii.a2b_base64({!r}))\r\n".format(chunk))
    write_text(ser, "f.close()\r\n")
    write_text(ser, "print('UPLOAD_OK')\r\n")
    ser.write(b"\x04")
    ser.flush()
    out = read_until_quiet(ser, quiet_s=0.3, timeout_s=4.0)
    if b"UPLOAD_OK" not in out:
        raise RuntimeError("upload confirmation missing: {}".format(out.decode("utf-8", "ignore")))


def soft_reset(ser):
    ser.write(b"\x04")
    ser.flush()
    time.sleep(0.4)
    return read_until_quiet(ser, quiet_s=0.3, timeout_s=3.0)


def main():
    parser = argparse.ArgumentParser(description="Upload a MicroPython file over serial without mpremote.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default="main.py")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        raise SystemExit("source file not found: {}".format(args.src))

    with serial.Serial(args.port, args.baud, timeout=0.15, write_timeout=1.0) as ser:
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass
        time.sleep(0.25)
        read_until_quiet(ser, timeout_s=0.6)
        upload_file(ser, args.src, args.dst)
        boot_output = soft_reset(ser)

    print("UPLOAD_OK")
    if boot_output:
        print(boot_output.decode("utf-8", "ignore").strip())


if __name__ == "__main__":
    main()
