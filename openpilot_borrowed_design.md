# OpenPilot-Inspired Airship Architecture (Laptop Brain + Pi Flight End)

Date: 2026-03-17

## Goal
Build an airship system that borrows openpilot's architecture and safety mindset while keeping all heavy computation on the laptop and real-time control on the Pi. The system must support:
- Manual flight control
- Natural-language command control
- Natural-language conversation
- Autonomous wandering when idle

This design does not reuse openpilot’s car-specific models. It borrows layered design, safety fallbacks, logging, and data-loop practices.

## High-Level Architecture

```
Sensors (Pi) --> Sense Uplink --> Laptop Perception/Planning --> Action Downlink --> Pi Control
```

### Layer 1: Real-Time Flight Control (Pi)
Responsibility: Closed-loop stability and immediate safety.
- Stabilization, altitude hold, motor output mapping.
- Executes high-level action commands from laptop (FORWARD/LEFT/UP/etc.).
- Handles link loss, invalid commands, or delays safely.

### Layer 2: Perception + Intent (Laptop)
Responsibility: High-latency tasks and semantic understanding.
- ASR (Whisper tiny or similar).
- LLM routing: command vs chat.
- VLM (optional / low-frequency).
- Autopilot wander generation when no user input.

### Layer 3: Safety + Arbitration (Laptop)
Responsibility: Decide which command source is valid at any moment.
Priority order:
1. Manual control
2. User command (NL)
3. Autopilot wander

## Data Interfaces

### Uplink (Pi -> Laptop)
Endpoint: `POST /api/sense`
Payload:
```
{
  "image": "<base64 jpg>",
  "audio": "<base64 wav>",
  "text": "<optional text override>"
}
```

### Downlink (Laptop -> Pi)
Response from `/api/sense`:
```
{
  "action": "FORWARD|LEFT|RIGHT|STOP|UP|DOWN|BACKWARD",
  "alt_setpoint": 0,
  "chat": "<optional reply text>"
}
```

## Timing & Latency Budget
- Control loop on Pi must run without network dependency.
- Laptop decisions may be slow; commands are *advisory* to Pi.
- Autopilot wander is low frequency (2–3s steps).
- Vision model (VLM) should be low frequency or optional.

## Safety Behavior

### Link Loss
If no valid command received within `X` seconds:
- Enter safe hover or slow descent mode (Pi-side).

### Invalid Commands
If laptop returns unknown command:
- Pi ignores and holds last safe state.

### Idle Mode
If no user input > `AUTOPILOT_IDLE_SEC`:
- Laptop sends wander commands.

## Model Routing Strategy (Two-Layer)

### Command Layer (Fast)
Small model + keyword fallback:
- Goal: parse command quickly.
- Output: flight action only.

### Chat Layer (Slow)
Larger model for quality:
- Only used if intent is chat.
- Output: text reply only.

## Logging & Replay
Inspired by openpilot’s logging:
- On laptop, save `/api/sense` payloads + actions.
- Use logs to replay and test model changes.
- Store in a rolling log file for post-flight analysis.

## Implementation Plan (Current Code)

### Laptop (`airpuff_server.py`)
- Two-layer LLM routing (command vs chat).
- Autopilot wander when idle.
- Keyword fallback for command reliability.
- Optional VLM with throttling.

### Pi (`airpuff_client.py`)
- Pure sensing + execution node.
- Hardware capture only (no model inference).
- Optional disable audio/video to reduce load.

## Future Improvements
- Add structured telemetry (IMU, altitude, battery).
- Add command rate limiting and smoothing.
- Add safety state machine on Pi.
- Add perception filters (simple optical flow) to reduce dependency on VLM.
