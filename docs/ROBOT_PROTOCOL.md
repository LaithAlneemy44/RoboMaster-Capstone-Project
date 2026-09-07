# Robot serial protocol

Contract between the driver-station GUI (`scripts/robot_gui.py`) and the robot's
microcontroller. Written for whoever implements the firmware side.

The authoritative encoder/decoder is `scripts/robot/protocol.py`; it has a selftest
(`python scripts/robot/protocol.py --selftest`). If this document and that module
disagree, the module is right and this document is a bug.

## Link

| | |
|---|---|
| Baud | 115200, 8N1 |
| Framing | one message per line, `\n` terminated |
| Encoding | ASCII |
| Flow control | none |

The host opens a named port. Whether that is a USB cable, a Bluetooth SPP pairing or a
2.4 GHz radio dongle is invisible to both ends and must stay that way.

Messages are human-readable on purpose: the entire protocol can be driven by hand from a
serial monitor, which is how you find out whether a fault is the GUI, the link, or the
firmware.

## Units

Drive and turret values are **normalised to -1.0 … 1.0**, never physical units. The host
does not know the gearing, wheel diameter or gimbal limits and should not; firmware maps
demand onto whatever the hardware can do. A motor swap then changes nothing on the host.

The host clamps every value before sending, and maps NaN to 0. Firmware may still range
check — trust, but verify.

## Host → robot

| Message | Meaning |
|---|---|
| `DRV <vx> <vy> <w>` | Chassis: forward, strafe, yaw rate. Each -1…1. |
| `GIM <dpitch> <dyaw>` | Turret **rate**, -1…1. Positive pitch is up, positive yaw is right. |
| `FIRE <n>` | Launch `n` projectiles, 1…10. Firmware owns cadence and any interlock. |
| `INTAKE <0\|1>` | Ball collection off/on. |
| `STOP` | All motors to zero, immediately. |
| `PING <seq>` | Heartbeat, `seq` wraps at 65536. |

`GIM` is a **rate**, not a position, because the host has no encoder feedback. Firmware
integrates it and enforces the mechanical limits.

## Robot → host

| Message | Meaning |
|---|---|
| `STA <batt> <pitch> <yaw> <flags>` | Battery 0…1, angles in degrees, `flags` bit field. |
| `ACK <seq>` | Optional acknowledgement. |
| `ERR <code> <text>` | Fault. `text` may contain spaces. |

`flags`: bit 0 armed, bit 1 intake running, bit 2 fault.

Send `STA` at about 10 Hz. The host uses its arrival — not its contents — to decide the
link is alive, so a steady `STA` is what keeps the operator's window live.

Anything unparseable is **ignored** by the host, never fatal. Serial links pick up noise
and a half-line at startup is normal.

## The heartbeat is a firmware requirement

**Firmware MUST stop all motion if it has received no message for 300 ms.**

This cannot be enforced from the host, which is exactly why it belongs here. The failure
it prevents is the important one: if the radio drops, the GUI crashes, or the operator's
laptop sleeps, a robot holding its last `DRV` command keeps driving. There is no message
that means "stop" in that scenario, because no message arrives at all. Silence has to be
the signal.

`scripts/robot/protocol.py` exports `HEARTBEAT_HZ = 10` and `TIMEOUT_MS = 300` so both
sides read the same numbers. The host sends `PING` at 10 Hz regardless of operator input,
giving three grace periods before the timeout bites.

Recovering from a timeout must require a fresh command. Do not resume the previous demand
when traffic returns.

## Reference exchange

```
host                          robot
  PING 1              ->
                      <-      STA 0.87 0.0 0.0 0
  DRV 0.600 0.000 0.000  ->
  GIM 0.000 -0.250    ->
                      <-      STA 0.86 0.0 -1.2 0
  PING 2              ->
  FIRE 1              ->
                      <-      ACK 2
  STOP                ->
```

## Minimal firmware skeleton

```c
uint32_t last_rx_ms = 0;

void loop() {
    if (read_line(buf)) {          // any complete line, including PING
        last_rx_ms = millis();
        dispatch(buf);
    }
    if (millis() - last_rx_ms > 300) {   // TIMEOUT_MS
        all_motors_off();                // silence means stop
    }
    every_100ms(send_status);
}
```
