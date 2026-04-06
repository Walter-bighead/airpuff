"""AirPuff ESP32-S3 serial bridge in MicroPython.

Stage 1 mirrors the Arduino stub protocol so the Pi and laptop stack can
be debugged without waiting for a full Arduino toolchain install.
"""

import sys
import time

try:
    import uselect as select
except ImportError:
    import select

try:
    import micropython

    micropython.kbd_intr(-1)
except Exception:
    pass


FAILSAFE_MS = 500
STATUS_PERIOD_MS = 250
MAX_LINE_LEN = 128
VALID_ACTIONS = {
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "STOP",
    "UP",
    "DOWN",
}

line_buffer = bytearray()
current_action = "STOP"
current_alt = 0
last_rx_ms = None
last_status_ms = 0
failsafe_active = True

stdin = getattr(sys.stdin, "buffer", sys.stdin)
stdout = sys.stdout
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)


def write_line(line):
    stdout.write(line + "\n")
    if hasattr(stdout, "flush"):
        stdout.flush()


def is_valid_action(action):
    return action in VALID_ACTIONS


def apply_outputs(action, alt):
    # Stage 1 only exposes the serial/control plumbing.
    _ = (action, alt)


def emit_status():
    now = time.ticks_ms()
    age_ms = 0 if last_rx_ms is None else time.ticks_diff(now, last_rx_ms)
    state = "FAILSAFE" if failsafe_active else "ACTIVE"
    write_line("STATE,{},{},{},{}".format(current_action, current_alt, age_ms, state))


def engage_failsafe():
    global current_action, current_alt, failsafe_active
    if failsafe_active and current_action == "STOP":
        return
    current_action = "STOP"
    current_alt = 0
    failsafe_active = True
    apply_outputs(current_action, current_alt)
    write_line("EVENT,FAILSAFE,STOP,0")


def accept_command(action, alt, ts_ms):
    global current_action, current_alt, last_rx_ms, failsafe_active
    current_action = action
    current_alt = alt
    last_rx_ms = time.ticks_ms()
    failsafe_active = False
    apply_outputs(current_action, current_alt)
    write_line("ACK,{},{},{}".format(current_action, current_alt, ts_ms))


def handle_line(line):
    if not line.startswith("AP,"):
        write_line("ERR,BAD_PREFIX,{}".format(line))
        return

    parts = line.split(",")
    if len(parts) != 4:
        write_line("ERR,BAD_FORMAT,{}".format(line))
        return

    _, action, alt_raw, ts_raw = parts
    if not is_valid_action(action):
        write_line("ERR,BAD_ACTION,{}".format(action))
        return

    try:
        alt = int(alt_raw)
        ts_ms = int(ts_raw)
    except ValueError:
        write_line("ERR,BAD_FORMAT,{}".format(line))
        return

    accept_command(action, alt, ts_ms)


def poll_serial():
    global line_buffer

    while poller.poll(0):
        chunk = stdin.read(1)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "ignore")
        byte = chunk[0]

        if byte == 10:
            line = bytes(line_buffer).decode("utf-8", "ignore")
            line_buffer = bytearray()
            handle_line(line)
        elif byte != 13:
            if len(line_buffer) >= MAX_LINE_LEN:
                line_buffer = bytearray()
                write_line("ERR,LINE_TOO_LONG")
            else:
                line_buffer.append(byte)


def poll_failsafe():
    global last_rx_ms
    if last_rx_ms is None:
        return
    if time.ticks_diff(time.ticks_ms(), last_rx_ms) > FAILSAFE_MS:
        engage_failsafe()
        last_rx_ms = None


def poll_status():
    global last_status_ms
    now = time.ticks_ms()
    if time.ticks_diff(now, last_status_ms) < STATUS_PERIOD_MS:
        return
    last_status_ms = now
    emit_status()


def main():
    write_line("AirPuff ESP32-S3 Bridge Ready")
    write_line("INFO,PROTO,AP,<ACTION>,<ALT>,<TS_MS>")
    emit_status()

    while True:
        poll_serial()
        poll_failsafe()
        poll_status()
        time.sleep_ms(10)


main()
