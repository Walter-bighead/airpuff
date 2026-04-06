"""AirPuff ESP32-S3 execution-layer skeleton in MicroPython.

This stage keeps the existing serial protocol stable while adding:
- a local flight state machine
- action-to-axis target mapping
- 4-channel mixer math
- optional PWM output channels for future ESC/servo wiring

By default all outputs stay disabled, so it remains safe on a desk.
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

try:
    import machine

    PWM = machine.PWM
    Pin = machine.Pin
except Exception:
    machine = None
    PWM = None
    Pin = None


FAILSAFE_MS = 500
STATUS_PERIOD_MS = 250
CONTROL_PERIOD_MS = 20
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


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class AxisTarget:
    def __init__(self, forward=0.0, yaw=0.0, vertical=0.0):
        self.forward = float(forward)
        self.yaw = float(yaw)
        self.vertical = float(vertical)


class OutputChannel:
    def __init__(self, name, pin=None, freq=50, neutral_us=1500, span_us=300):
        self.name = name
        self.pin_num = pin
        self.freq = int(freq)
        self.neutral_us = int(neutral_us)
        self.span_us = int(span_us)
        self.enabled = False
        self.last_value = 0.0
        self._pwm = None

        if pin is not None and PWM is not None and Pin is not None:
            try:
                self._pwm = PWM(Pin(pin), freq=self.freq)
                self.write(0.0)
                self.enabled = True
            except Exception:
                self._pwm = None
                self.enabled = False

    def write(self, value):
        value = clamp(float(value), -1.0, 1.0)
        self.last_value = value
        if not self.enabled or self._pwm is None:
            return

        pulse_us = int(self.neutral_us + (self.span_us * value))
        pulse_us = clamp(pulse_us, 900, 2100)
        period_us = int(1000000 / self.freq)

        if hasattr(self._pwm, "duty_ns"):
            self._pwm.duty_ns(pulse_us * 1000)
        elif hasattr(self._pwm, "duty_u16"):
            duty = int((pulse_us / period_us) * 65535)
            self._pwm.duty_u16(clamp(duty, 0, 65535))


class OutputBackend:
    def __init__(self):
        # Leave pins as None until the real ESC/servo wiring is fixed.
        self.channels = [
            OutputChannel("front_left", pin=None),
            OutputChannel("front_right", pin=None),
            OutputChannel("rear_left", pin=None),
            OutputChannel("rear_right", pin=None),
        ]

    def apply(self, mixed_values):
        for channel, value in zip(self.channels, mixed_values):
            channel.write(value)

    def describe(self):
        enabled = 0
        names = []
        for channel in self.channels:
            if channel.enabled:
                enabled += 1
            names.append("{}:{}".format(channel.name, channel.pin_num))
        return enabled, "|".join(names)

    def snapshot(self):
        return ",".join("{:.2f}".format(ch.last_value) for ch in self.channels)


class AirshipMixer:
    def mix(self, target):
        values = [
            target.vertical + target.forward - target.yaw,
            target.vertical + target.forward + target.yaw,
            target.vertical - target.forward - target.yaw,
            target.vertical - target.forward + target.yaw,
        ]
        peak = max(1.0, max(abs(v) for v in values))
        return [clamp(v / peak, -1.0, 1.0) for v in values]


class FlightController:
    def __init__(self):
        self.state = "BOOT"
        self.current_action = "STOP"
        self.current_alt = 0
        self.last_rx_ms = None
        self.last_status_ms = 0
        self.last_control_ms = 0
        self.target = AxisTarget()
        self.backend = OutputBackend()
        self.mixer = AirshipMixer()
        self.last_mixed = [0.0, 0.0, 0.0, 0.0]

    def set_action(self, action, alt):
        self.current_action = action
        self.current_alt = int(alt)
        self.last_rx_ms = time.ticks_ms()
        self.state = "ACTIVE"
        self.target = self._target_from_action(action, alt)

    def _target_from_action(self, action, alt):
        _ = alt
        if action == "FORWARD":
            return AxisTarget(forward=0.45)
        if action == "BACKWARD":
            return AxisTarget(forward=-0.45)
        if action == "LEFT":
            return AxisTarget(yaw=-0.35)
        if action == "RIGHT":
            return AxisTarget(yaw=0.35)
        if action == "UP":
            return AxisTarget(vertical=0.35)
        if action == "DOWN":
            return AxisTarget(vertical=-0.35)
        return AxisTarget()

    def engage_failsafe(self):
        if self.state == "FAILSAFE" and self.current_action == "STOP":
            return False
        self.current_action = "STOP"
        self.current_alt = 0
        self.state = "FAILSAFE"
        self.target = AxisTarget()
        self.last_mixed = self.mixer.mix(self.target)
        self.backend.apply(self.last_mixed)
        return True

    def poll_control(self):
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_control_ms) < CONTROL_PERIOD_MS:
            return
        self.last_control_ms = now
        self.last_mixed = self.mixer.mix(self.target)
        self.backend.apply(self.last_mixed)

    def poll_failsafe(self):
        if self.last_rx_ms is None:
            return False
        if time.ticks_diff(time.ticks_ms(), self.last_rx_ms) > FAILSAFE_MS:
            self.last_rx_ms = None
            return self.engage_failsafe()
        return False

    def should_emit_status(self):
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_status_ms) < STATUS_PERIOD_MS:
            return False
        self.last_status_ms = now
        return True

    def status_line(self):
        now = time.ticks_ms()
        age_ms = 0 if self.last_rx_ms is None else time.ticks_diff(now, self.last_rx_ms)
        state = "FAILSAFE" if self.state == "FAILSAFE" else "ACTIVE"
        return "STATE,{},{},{},{}".format(self.current_action, self.current_alt, age_ms, state)

    def control_line(self):
        return "STATECTL,{},{:.2f},{:.2f},{:.2f},{}".format(
            self.state,
            self.target.forward,
            self.target.yaw,
            self.target.vertical,
            self.backend.snapshot(),
        )


controller = FlightController()
line_buffer = bytearray()
stdin = getattr(sys.stdin, "buffer", sys.stdin)
stdout = sys.stdout
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)


def write_line(line):
    stdout.write(line + "\n")
    if hasattr(stdout, "flush"):
        stdout.flush()


def sanitize_line(line):
    chars = []
    for ch in line:
        code = ord(ch)
        if ch in {",", "-", "_", ".", " "} or ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            chars.append(ch)
        elif code >= 32:
            chars.append(ch)
    return "".join(chars).strip()


def is_valid_action(action):
    return action in VALID_ACTIONS


def accept_command(action, alt, ts_ms):
    controller.set_action(action, alt)
    write_line("ACK,{},{},{}".format(controller.current_action, controller.current_alt, ts_ms))


def handle_line(line):
    line = sanitize_line(line)
    if not line:
        return
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


def main():
    enabled, pin_desc = controller.backend.describe()
    controller.engage_failsafe()
    write_line("AirPuff ESP32-S3 Exec Ready")
    write_line("INFO,PROTO,AP,<ACTION>,<ALT>,<TS_MS>")
    write_line("INFO,OUTPUTS,enabled={},pins={}".format(enabled, pin_desc))
    write_line(controller.status_line())
    write_line(controller.control_line())

    while True:
        poll_serial()
        controller.poll_control()
        if controller.poll_failsafe():
            write_line("EVENT,FAILSAFE,STOP,0")
        if controller.should_emit_status():
            write_line(controller.status_line())
            write_line(controller.control_line())
        time.sleep_ms(5)


main()
