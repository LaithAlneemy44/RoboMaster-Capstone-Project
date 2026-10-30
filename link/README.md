# Pico WiFi link

Wireless command link from the MacBook to the robot's Arduino Mega, through two Raspberry Pi
Pico 2 WH boards. The MacBook never connects to the robot by wire.

```
MacBook ──USB serial──▶ LAPTOP Pico ···WiFi UDP···▶ ROBOT Pico ──UART──▶ Arduino Mega
        ◀──────────────               ◀···········              ◀──────   (telemetry, optional)
                        (station)                    (access point)       Serial1, pin 19
```

| Setting | Value |
| --- | --- |
| Network | ROBOT Pico hosts the access point `robo-link` at `192.168.4.1`; LAPTOP joins it |
| Transport | UDP, port 4210, one line per datagram. Both Picos relay lines verbatim, so the link carries any newline-framed ASCII protocol |
| Return path | Lines the receiver sends on ROBOT's GP1 go back to the sender of the last command, and LAPTOP writes them to USB serial |
| Serial | USB serial MacBook → LAPTOP; UART 115200 8N1 ROBOT → Mega |
| Failsafe | ROBOT sends `S\n` to the Mega after 300 ms with no packet; the Mega also stops on its own after 500 ms of silence on Serial1 |

## Commands

Newline-terminated ASCII:

```
A,<seq>,<yaw>,<pitch>,<fire>     e.g. A,1234,12.50,-3.20,1
S                                stop
```

- `seq`: increasing counter, used by the Mega to detect lost packets.
- `yaw`, `pitch`: aim angles in degrees.
- `fire`: `1` fires, `0` doesn't.

**This replaces the draft protocol on this link.** `docs/ROBOT_PROTOCOL.md` and
`scripts/robot/protocol.py` (`DRV`/`GIM`/`FIRE`/`STOP`/`PING`, turret *rates*, 300 ms stop in
firmware) were written for a single wired hop and have not been changed to match. `driver.py`,
`aiming.py` and the GUI still speak the old format. Reconciling the two, including angles versus
rates, is still open.

**Both formats have now crossed the link.** The relay doesn't parse lines, so this is a
receiver decision, not a link one. On 2026-10-28 the unchanged driver-station GUI
(`scripts/robot_gui.py --driver serial`) drove `mkr_station_test` in the GUI's own format,
with its STA telemetry coming back. That shows the GUI's format works over the relay; it
doesn't choose it.

## Files

| File | Runs on | Role |
| --- | --- | --- |
| `robot_pico_main.py` | ROBOT Pico, as `main.py` | Access point, UDP → UART, UART replies → UDP, 300 ms failsafe, AP self-restart |
| `laptop_pico_main.py` | LAPTOP Pico, as `main.py` | Joins the AP, USB serial lines → UDP, UDP replies → USB serial |
| `flash_test_gui.py` + `mkr_flash_test/` | MacBook + MKR Zero | Key presses → LED patterns, one-way |
| `mkr_station_test/mkr_station_test.ino` | MKR Zero | Speaks the GUI's protocol and answers with STA, so the real GUI runs against it |
| `link_secrets.py` | both Picos | WiFi password. **Gitignored**; create it from `link_secrets.example.py` |
| `mac_link_test.py` | MacBook | Streams fake aim commands at 50 Hz, for testing without the vision pipeline |
| `mega_receiver/mega_receiver.ino` | Arduino Mega | Parses commands, echoes them to USB Serial, warns on skipped `seq` |
| `firmware/` | — | MicroPython `.uf2`, downloaded, gitignored |

## Wiring (ROBOT Pico ↔ Mega)

| Pico 2 WH | Mega 2560 | Notes |
| --- | --- | --- |
| Pin 1, GP0 (UART0 TX) | Pin 19 (RX1) | Direct wire. The Pico's 3.3 V high clears the Mega's 3.0 V input threshold |
| Pin 3, GND | GND | **Common ground is required** |
| Pin 2, GP1 (UART0 RX) | — | Return path. Unconnected on the Mega for now (see below) |

- **If Mega → Pico is ever needed**, Mega TX1 (pin 18) must go through a divider before Pico
  GP1: 1 kΩ in series, then 2 kΩ to ground, giving about 3.3 V. **Pico pins are not 5 V
  tolerant**, and a direct wire will damage the Pico.
- **Power for the ROBOT Pico:**
  - testing: a USB power bank, or a charge-only USB cable (it needs power only, not data)
  - on the robot: the 5 V rail into VSYS (pin 39), with ground shared
- **Use `Serial1` on the Mega, not `Serial`.** Pins 0 and 1 are shared with its USB port.

## Setup

### Tools (MacBook)

```sh
pip3 install mpremote pyserial
```

### Password

```sh
cp link/link_secrets.example.py link/link_secrets.py   # then edit PASSWORD, at least 8 characters
```

### Flash MicroPython (each Pico, once)

1. Download the **RPI_PICO2_W** build, not PICO_W, from
   <https://micropython.org/download/RPI_PICO2_W/> into `link/firmware/`. Tested with v1.29.0.
2. Hold **BOOTSEL** while plugging the Pico into the Mac, then release. A drive called
   `RP2350` appears. A brand-new Pico shows it without BOOTSEL.
3. `cp link/firmware/RPI_PICO2_W-*.uf2 /Volumes/RP2350/`. The drive disappears and the Pico
   reboots into MicroPython. A "Disk Not Ejected Properly" warning is harmless.

BOOTSEL is only for flashing. Don't hold it on normal power-ups, or `main.py` won't run.

### Load the scripts

Find the port with `ls /dev/cu.usbmodem*`. Use `cu.`, not `tty.`.

```sh
cd link
# ROBOT
mpremote connect /dev/cu.usbmodemXXXX cp link_secrets.py :link_secrets.py + cp robot_pico_main.py :main.py
# LAPTOP
mpremote connect /dev/cu.usbmodemXXXX cp link_secrets.py :link_secrets.py + cp laptop_pico_main.py :main.py
```

Both files must be on each Pico; without `link_secrets.py`, `main.py` fails at its import.
Label the boards. To tell which is which over USB, read the chip ID:

```sh
mpremote connect /dev/cu.usbmodemXXXX exec "import machine; print(machine.unique_id().hex())"
```

| Board | Chip ID | `main.py` |
| --- | --- | --- |
| ROBOT | `ad6515c01640a363` | `robot_pico_main.py` |
| LAPTOP | `6e433c0184bf5a76` | `laptop_pico_main.py` |

The roles were swapped on 2026-10-29 and swapped back on 2026-10-30.

### Mega

Open `mega_receiver/mega_receiver.ino` in the Arduino IDE, select **Arduino Mega or Mega
2560**, and upload. Serial Monitor at 115200 shows every parsed command. Fill in the
`PLACEHOLDER` functions (`setYaw`, `setPitch`, `setFire`, `stopAll`) with the real actuator
code. They must return immediately: no `delay()`.

From the terminal instead (`brew install arduino-cli`, plus `arduino-cli core install
arduino:avr` if the Arduino IDE isn't installed), run from `link/`:

```sh
arduino-cli compile --fqbn arduino:avr:mega mega_receiver
arduino-cli upload  --fqbn arduino:avr:mega -p /dev/cu.usbmodemXXXX mega_receiver
```

## Desk test

### LEDs

| Board | LED | Meaning |
| --- | --- | --- |
| ROBOT | steady on | access point is up, no packets yet |
| ROBOT | flickering | packets arriving; flips at most every 200 ms, so any steady stream visibly flickers |
| LAPTOP | off | trying to join `robo-link` |
| LAPTOP | steady on | joined |
| Mega | flickering (pin 13, `L`) | valid `A` commands arriving on Serial1 |
| Mega | off | no commands yet, or stopped (link `S` or its own 500 ms timeout) |

### With one data cable

Only LAPTOP needs data. Power ROBOT from a charge-only cable or power bank.

1. Plug in ROBOT. Its LED comes on steady within a few seconds.
2. Plug in LAPTOP on the data cable. Its LED comes on steady once joined.
3. `python3 link/mac_link_test.py /dev/cu.usbmodemXXXX`, then Ctrl-C after a few seconds.
4. **Pass:** ROBOT's LED flickers while the test runs and stops when it ends.

This proves the full path but measures nothing, because ROBOT's output can't be read.

### Measuring rate, loss and failsafe

Give ROBOT the data cable instead, and have LAPTOP, on a charge-only cable, generate its own
50 Hz commands with a temporary test `main.py` (restore `laptop_pico_main.py` afterwards).
With two data cables, run `mac_link_test.py` on LAPTOP as normal instead. Either way, capture
ROBOT's USB output during the test (pyserial, or
`mpremote connect <port> repl` under a timeout; don't leave a REPL open). **Pass:**

- `AP up: ('192.168.4.1', ...)` at boot
- `-> b'A,...'` lines streaming during the test
- `failsafe stop` about 300 ms after the test stops

Packet rate and loss come from counting `->` lines and gaps in `seq`.

### Results so far (2026-10-18, boards side by side)

- **Signal:** −14 to −27 dBm RSSI, channel 3. LAPTOP joins in about 5 s.
- **Loss:** 51/51 packets received in a 10 s window at 5 Hz, `seq` 140–190 with no gaps.
- **Full path at 50 Hz:** MacBook → LAPTOP → ROBOT confirmed by ROBOT's LED.
- **WiFi hop at 50 Hz, measured from ROBOT's output:** 1500/1500 packets in 30 s, 50.1
  packets/s, `seq` 2163–3662 with **no gaps (0% loss)**. LAPTOP ran a temporary script
  generating `mac_link_test.py`-format commands at 50 Hz itself, so that ROBOT could have the
  only data cable. The MacBook → LAPTOP USB hop is not in this number.
- **Failsafe:** with LAPTOP unplugged mid-stream, ROBOT printed `failsafe stop` once, 271 ms
  after the last packet as timestamped on the Mac. The Pico's timer is 300 ms, and USB output
  delivery skews the Mac's timestamps.
- **Recovery:** ROBOT was unplugged and replugged with LAPTOP left running. LAPTOP's LED went
  off, came back on by itself, and commands reached ROBOT again with no restart of LAPTOP. A
  soft reset of LAPTOP also recovers.
- **Mega sketch:** compiles cleanly for the Mega 2560 with all warnings on (arduino-cli
  1.5.1, arduino:avr 1.8.8): 6.9 KB flash (3%), 459 B RAM (5%). Its parser was also tested
  on the Mac against valid, malformed, overlong and out-of-order lines. It has not run on a
  Mega.
- **Not yet tested:** range, loss with the robot's motors running, and the Mega receiving
  real commands on Serial1.

### Results 2026-10-21 (MKR Zero standing in for the Mega)

- **Full path to a receiver works:** MacBook → LAPTOP → WiFi → ROBOT → UART → MKR Zero.
  `flash_test_gui.py` key presses played the right LED pattern on the MKR, with the ROBOT
  Pico powered from the MKR's 5V pin into VSYS.
- **AP stall, once:** after a few minutes of testing, ROBOT kept beaconing `robo-link` but
  LAPTOP's joins failed (status −1 within 3 s, signal −22 dBm). Only a ROBOT power cycle
  cleared it. Cause unknown; a power dip through a push-on jumper is suspected.
- **Self-restart added to `robot_pico_main.py`:** it restarts the AP after 120 s with no
  packets, or 20 s with no associated station.
  - The 120 s rule **fired as designed** (16:28:02, AP back in 1 s).
  - The 20 s rule **did not fire** within ~60 s of unplugging LAPTOP, although
    `ap.status("stations")` returned `[]` when queried afterwards. **Unverified; debug next.**
  - LAPTOP's LED did not go off when the AP restarted, so LAPTOP may keep a stale
    association after an AP restart. Check that it actually rejoins.

### Flash test with a spare board (no Mega)

`flash_test_gui.py` is a stripped-down copy of the driver-station GUI. Each key press sends one
`L,<seq>,<code>` line through the link, and `mkr_flash_test/mkr_flash_test.ino` (Arduino MKR
Zero) plays a matching pattern on its built-in LED. Keys: `1`-`5` blink that many times, `F`
fast flicker, `H` hold on 1 s, `0`/Esc off. It tests the link only: no movement commands.

- **Wiring:** Pico GP0 (pin 1) → MKR pin 13 (RX), and GND → GND. Both boards are 3.3 V.
- **Upload:** `arduino-cli upload --fqbn arduino:samd:mkrzero -p /dev/cu.usbmodemXXXX mkr_flash_test`
  (`arduino-cli core install arduino:samd` once). It blinks 3 times quickly on boot.
- **The port vanishing after an upload** was macOS missing the board's microsecond-long USB
  drop on a software reset; it came back only on a replug. The sketch now detaches from USB for
  500 ms after any non-power-on reset, and the port returned unaided on 3/3 uploads
  (2026-10-21, through the USB-C multiport adapter's hub). Other SAMD sketches need the same.
- **Run:** `.venv/bin/python link/flash_test_gui.py`, pick LAPTOP's port, then optionally the
  MKR's port as the receiver, to see what it parsed in the same window.
- The ROBOT Pico sends `S` 300 ms after every key press (its failsafe). The MKR only logs it,
  or every pattern would be cut short.

### Real GUI against an MKR Zero (no Mega)

`mkr_station_test` lets the unchanged driver-station GUI run over the link with nothing that
moves. The GUI blocks drive, turret and fire whenever telemetry is silent for 1 s, so this test
needs the return path.

- **Wiring (4 jumpers):** Pico GP0 (pin 1) → MKR 13 RX; MKR 14 TX → Pico GP1 (pin 2); MKR 5V →
  Pico VSYS (pin 39); GND → GND. Both boards are 3.3 V, so no divider.
- **Cables:** power → MKR (it powers ROBOT through VSYS), data → LAPTOP, camera → Mac.
- **Run:** `.venv/bin/python scripts/robot_gui.py --driver serial --port /dev/cu.usbmodemXXXX
  --source 0 --weights runs/detect/fast_640/weights/best.pt --imgsz 640`
- **MKR LED:** on while any drive or turret key is held; 3 quick flashes on FIRE.
- The MKR answers with STA only while PINGs arrive, so "link: OK" means the round trip works.
  It has no battery sensor, so the GUI shows 0% battery.

### Results 2026-10-28 (real GUI, C270, MKR Zero)

- **Round trip measured at the serial level:** 51 STA lines in 5 s (10 Hz), the first 0.1 s
  after the first PING. One second of `GIM 0.5 0` moved the reported pitch from 0° to 9°,
  so commands reach the receiver and its replies reflect them.
- **The real GUI worked end to end** (user-verified): LIVE banner, link OK, camera image, and
  the MKR LED following the keys.
- **The C270 opened as index 0 at 1280×960**, a 4:3 mode, not the 1280×720 the handoff
  expects. The training footage is 16:9, so check what frame the detector actually gets.

### Mega with the real GUI (`mega_station_test`), 2026-10-29: in progress

Same test as the MKR, on the Mega: every GUI key gives a different flash count on the Mega's
`L` LED (table at the top of the sketch). The Mega's TX1 is 5 V, so it reaches the Pico
through a divider of three 1 kΩ resistors, giving about 3.3 V:

```
Mega TX1 (18) ──[1 kΩ]──●──[1 kΩ]──[1 kΩ]── GND (Mega GND + Pico GND)
                        └── Pico GP1 (pin 2)
Pico GP0 (pin 1) ── Mega RX1 (19)      direct
Mega 5V          ── Pico VSYS (pin 39)  powers the robot Pico
```

- **Built on the robot's breadboard** in columns 20/25/30/35, away from the Pico (columns
  41–60). The printed column numbers on that board sit about one hole off; count holes.
- **The Mega side works:** commands sent over its own USB (the sketch accepts both) were all
  parsed: W, D, I, L, FIRE.
- **Nothing reached the Mega.** Its own log showed no commands and no STA came back, with
  LAPTOP joined to `robo-link` (−57 dBm, also after a fresh rejoin).
- ~~ROBOT's LED stayed steady, so ROBOT wasn't receiving.~~ **Wrong, see 2026-10-30.** The
  LED couldn't show that.

**2026-10-30, narrowing it down without the Mega:**
- **The robot script doesn't crash.** On USB power, `6e433c…` booted, brought the AP up and sat
  in its main loop when interrupted.
- **The Pico-to-Pico link works.** With roles swapped back (`ad6515…` as ROBOT on USB, LAPTOP
  on a charge-only cable running a temporary self-sending script), ROBOT received 321 of 322
  datagrams in 30 s with no `seq` gaps.
- **The "steady LED" evidence was an artifact.** ROBOT flipped its LED on every packet, and
  every failing test sent PING and GIM back to back, so each pair flipped it off and straight
  back on, too fast to see. A busy link looked identical to a dead one. Every test that sent
  single commands flickered, including Mac → LAPTOP script → ROBOT with ROBOT powered through
  jumpers from LAPTOP's VBUS. Fixed in `robot_pico_main.py`: the LED now flips at most every
  200 ms. Flashed onto ROBOT (`ad6515…`) on 2026-10-30; it boots and brings the AP up.
- **So the relay probably worked on 2026-10-29 too, and the break was after ROBOT:** the
  GP0 → Mega RX1 (19) wire or its breadboard contact, or the Mega's Serial1 input, which was
  never tested on its own (only its USB input was). Next test: ROBOT's USB output and the
  Mega's log read at the same time, to see where commands stop. Don't diagnose from the LED.
- **Loopback, 2026-10-30: everything up to ROBOT's pins works.** ROBOT GP0 (pin 1) jumpered
  to its own GP1 (pin 2), ROBOT on a charge-only cable, LAPTOP on the Mac. 145 of 145 `PING`s
  came back to the Mac, plus ROBOT's own failsafe `S`. That covers Mac → LAPTOP → WiFi →
  ROBOT → UART TX → UART RX → ROBOT → WiFi → LAPTOP → Mac. **The 2026-10-29 break was
  between ROBOT's pins and the Mega:** the Pico's contact in the robot breadboard, a wire one
  column or one Mega hole off, or (least likely) the Mega's Serial1.
- `laptop_pico_main.py` no longer swallows `sendto` errors: it prints `send failed: …` and
  rejoins with a fresh socket. No error has appeared yet; it was added while chasing this.
- **The Mega is a generic Mega 2560 R3** (ATmega2560 + 16U2). The official A000067 pinout
  applies, but its TX/RX table has typos; trust the board's silkscreen for TX1 18 / RX1 19.
  Its 3.3V pin gives 50 mA, too little for the Pico.

## Gotchas

- **LAPTOP's USB serial is also its MicroPython REPL.** Send only plain ASCII lines. A `0x03`
  byte is Ctrl-C and kills `main.py`, and `0x04` soft-resets it.
- **LAPTOP rejoins by itself if ROBOT restarts**, for example after a power blip on the
  robot. Its LED goes off while ROBOT is down and comes back on within about 10–20 s of
  ROBOT's network returning, with no replug. `connect()` drops any stale association before
  each attempt and retries every 10 s. Before that fix, LAPTOP never rejoined, and after a
  soft reset (Ctrl-D) it could report itself joined while nothing reached ROBOT. Both cases
  were tested and pass (2026-10-18).
- If `mpremote` can't connect because `main.py` is busy, use
  `mpremote connect <port> exec "..."`. It interrupts `main.py` first.
- WiFi power saving is disabled on both Picos for latency.
- A charge-only micro-USB cable powers a Pico but never shows up on the Mac. If neither a
  `RP2350` drive nor a `/dev/cu.usbmodem*` appears, try another cable first.
