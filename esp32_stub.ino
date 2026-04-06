// AirPuff ESP32-S3 bridge skeleton
// Board target: ESP32S3 Dev Module (ESP32-S3-N16R8 class board)
// This stage focuses on serial bring-up, failsafe, and control plumbing.

#include <Arduino.h>

namespace {

constexpr unsigned long kSerialBaud = 115200;
constexpr unsigned long kFailsafeMs = 500;
constexpr unsigned long kStatusPeriodMs = 250;

String lineBuffer;
String currentAction = "STOP";
int currentAlt = 0;
unsigned long lastRxMs = 0;
unsigned long lastStatusMs = 0;
bool failsafeActive = true;

bool isValidAction(const String &action) {
  return action == "FORWARD" || action == "BACKWARD" || action == "LEFT" ||
         action == "RIGHT" || action == "STOP" || action == "UP" ||
         action == "DOWN";
}

void applyOutputs(const String &action, int alt) {
  // Stage 1: only expose plumbing and debug info.
  // Stage 2: this function will map high-level commands to ESC outputs.
  (void)action;
  (void)alt;
}

void emitStatus() {
  const unsigned long now = millis();
  const unsigned long ageMs = (lastRxMs == 0) ? 0 : (now - lastRxMs);
  Serial.print("STATE,");
  Serial.print(currentAction);
  Serial.print(",");
  Serial.print(currentAlt);
  Serial.print(",");
  Serial.print(ageMs);
  Serial.print(",");
  Serial.println(failsafeActive ? "FAILSAFE" : "ACTIVE");
}

void engageFailsafe() {
  if (failsafeActive && currentAction == "STOP") {
    return;
  }
  currentAction = "STOP";
  currentAlt = 0;
  failsafeActive = true;
  applyOutputs(currentAction, currentAlt);
  Serial.println("EVENT,FAILSAFE,STOP,0");
}

void acceptCommand(const String &action, int alt, unsigned long tsMs) {
  currentAction = action;
  currentAlt = alt;
  lastRxMs = millis();
  failsafeActive = false;
  applyOutputs(currentAction, currentAlt);

  Serial.print("ACK,");
  Serial.print(currentAction);
  Serial.print(",");
  Serial.print(currentAlt);
  Serial.print(",");
  Serial.println(tsMs);
}

void handleLine(const String &line) {
  // Expected format: AP,<ACTION>,<ALT>,<TS_MS>
  if (!line.startsWith("AP,")) {
    Serial.print("ERR,BAD_PREFIX,");
    Serial.println(line);
    return;
  }

  const int p1 = line.indexOf(',', 3);
  const int p2 = (p1 >= 0) ? line.indexOf(',', p1 + 1) : -1;
  const int p3 = (p2 >= 0) ? line.indexOf(',', p2 + 1) : -1;
  if (p1 < 0 || p2 < 0 || p3 < 0) {
    Serial.print("ERR,BAD_FORMAT,");
    Serial.println(line);
    return;
  }

  const String action = line.substring(3, p1);
  const int alt = line.substring(p1 + 1, p2).toInt();
  const unsigned long tsMs = strtoul(line.substring(p2 + 1, p3).c_str(), nullptr, 10);

  if (!isValidAction(action)) {
    Serial.print("ERR,BAD_ACTION,");
    Serial.println(action);
    return;
  }

  acceptCommand(action, alt, tsMs);
}

void pollSerial() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      handleLine(lineBuffer);
      lineBuffer = "";
    } else if (c != '\r') {
      lineBuffer += c;
      if (lineBuffer.length() > 128) {
        lineBuffer = "";
        Serial.println("ERR,LINE_TOO_LONG");
      }
    }
  }
}

void pollFailsafe() {
  const unsigned long now = millis();
  if (lastRxMs == 0) {
    return;
  }
  if ((now - lastRxMs) > kFailsafeMs) {
    engageFailsafe();
    lastRxMs = 0;
  }
}

void pollStatus() {
  const unsigned long now = millis();
  if ((now - lastStatusMs) < kStatusPeriodMs) {
    return;
  }
  lastStatusMs = now;
  emitStatus();
}

}  // namespace

void setup() {
  Serial.begin(kSerialBaud);
  delay(200);
  Serial.println("AirPuff ESP32-S3 Bridge Ready");
  Serial.println("INFO,PROTO,AP,<ACTION>,<ALT>,<TS_MS>");
  emitStatus();
}

void loop() {
  pollSerial();
  pollFailsafe();
  pollStatus();
}
