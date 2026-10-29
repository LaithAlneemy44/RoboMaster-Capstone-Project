// Arduino Mega 2560 LED test for the real driver-station GUI over the Pico WiFi link.
// Nothing here drives a motor: every key press in the GUI shows up as a flash count on the
// Mega's built-in "L" LED (pin 13), and STA telemetry goes back so the GUI sees a live link.
//
// The GUI sends commands, not key names (scripts/robot/protocol.py), so the key is worked
// out from which value just became non-zero. A key plays its pattern once, when pressed.
//
//   Key        Command                 LED
//   W / S      DRV vx > 0 / < 0        1 / 2 short flashes
//   A / D      DRV vy < 0 / > 0        3 / 4 short flashes
//   ← / →      DRV w  < 0 / > 0        5 / 6 short flashes
//   I / K      GIM pitch > 0 / < 0     1 / 2 long flashes
//   J / L      GIM yaw   < 0 / > 0     3 / 4 long flashes
//   Space      FIRE (only when ARMED)  fast flicker, 1 s
//   F          INTAKE 1 / 0            one 1.5 s glow
//   Esc        STOP                    LED off, pattern cancelled
//
// Every command that changes something is also printed on the USB Serial monitor (115200).
// The same commands typed into that monitor are obeyed too, so the LED patterns can be
// checked with the Mega alone, before the link is involved.
//
// Telemetry: "STA <battery> <pitch> <yaw> <flags>" at 10 Hz on Serial1 TX, only while PINGs
// keep arriving, so "link: OK" in the GUI means the round trip works. No battery sensor, so
// battery is reported as 0.
//
// Wiring:
//   Pico GP0 (pin 1)  -> Mega pin 19 (RX1)            direct; 3.3 V reads as high on the Mega
//   Mega pin 18 (TX1) -> divider -> Pico GP1 (pin 2)  5 V must come down to ~3.3 V first
//   Mega GND          -> Pico GND                      common ground is required
// See link/README.md for the divider.

#include <Arduino.h>

const unsigned long LINK_BAUD    = 115200;
const unsigned long DEBUG_BAUD   = 115200;
const unsigned long TIMEOUT_MS   = 300;    // protocol.TIMEOUT_MS: stop after this much silence
const unsigned long STA_EVERY_MS = 100;    // 10 Hz, matching the GUI's heartbeat
const float GIM_DEG_PER_TICK = 2.0;        // same integration as driver.MockDriver

const unsigned int SHORT_ON = 150, SHORT_OFF = 200;
const unsigned int LONG_ON = 500, LONG_OFF = 300;

const uint8_t MAX_LINE_LEN = 64;

struct LineReader {
  char buf[MAX_LINE_LEN + 1];
  uint8_t len;
  bool overflow;
};
LineReader linkReader = {{0}, 0, false};
LineReader usbReader = {{0}, 0, false};

float drv[3] = {0, 0, 0};       // vx, vy, w
float gimRate[2] = {0, 0};      // pitch, yaw
float pitchDeg = 0, yawDeg = 0;
bool intakeOn = false;

unsigned long lastRxMs = 0;
unsigned long lastPingMs = 0;
bool havePing = false;
bool stopped = true;
unsigned long nextStaMs = 0;

// Pattern player: `flashesLeft` on/off pairs, non-blocking.
uint8_t flashesLeft = 0;
unsigned int onMs = 0, offMs = 0;
bool ledOn = false;
unsigned long nextEdgeMs = 0;

void play(uint8_t count, unsigned int on, unsigned int off) {
  flashesLeft = count;
  onMs = on;
  offMs = off;
  ledOn = true;
  digitalWrite(LED_BUILTIN, HIGH);
  nextEdgeMs = millis() + on;
}

void cancelPattern() {
  flashesLeft = 0;
  ledOn = false;
  digitalWrite(LED_BUILTIN, LOW);
}

void updateLed() {
  if (flashesLeft == 0 || (long)(millis() - nextEdgeMs) < 0) return;
  if (ledOn) {
    ledOn = false;
    digitalWrite(LED_BUILTIN, LOW);
    flashesLeft--;
    nextEdgeMs += offMs;
  } else {
    ledOn = true;
    digitalWrite(LED_BUILTIN, HIGH);
    nextEdgeMs += onMs;
  }
}

// Plays `pos` flashes if `now` turned positive, `neg` if it turned negative.
// Returns true if it played something.
bool pressed(float before, float now, uint8_t pos, uint8_t neg,
             unsigned int on, unsigned int off, const __FlashStringHelper *name) {
  bool wasPos = before > 0, wasNeg = before < 0;
  if (now > 0 && !wasPos) {
    play(pos, on, off);
  } else if (now < 0 && !wasNeg) {
    play(neg, on, off);
  } else {
    return false;
  }
  Serial.print(F("key: "));
  Serial.print(name);
  Serial.println(now > 0 ? F(" +") : F(" -"));
  return true;
}

void allStop(const __FlashStringHelper *why) {
  drv[0] = drv[1] = drv[2] = 0;
  gimRate[0] = gimRate[1] = 0;
  intakeOn = false;
  cancelPattern();
  if (!stopped) {
    Serial.print(F("STOP ("));
    Serial.print(why);
    Serial.println(F(")"));
  }
  stopped = true;
}

// Parses exactly `n` space-separated floats from `rest`, clamped to -1..1.
bool parseFloats(char *rest, float *out, uint8_t n) {
  uint8_t got = 0;
  for (char *tok = strtok(rest, " "); tok != NULL; tok = strtok(NULL, " ")) {
    if (got == n) return false;
    char *end;
    double v = strtod(tok, &end);
    if (*end != '\0') return false;
    out[got++] = constrain(v, -1.0, 1.0);
  }
  return got == n;
}

void handleLine(char *line) {
  char *rest = strchr(line, ' ');
  if (rest) *rest++ = '\0';
  else rest = line + strlen(line);

  if (strcmp(line, "PING") == 0) {
    lastPingMs = millis();
    havePing = true;
    return;                                   // 10 Hz: too frequent to echo
  }
  if (strcmp(line, "STOP") == 0 || strcmp(line, "S") == 0) {
    allStop(strcmp(line, "S") == 0 ? F("link failsafe") : F("GUI"));
    return;
  }
  stopped = false;
  if (strcmp(line, "DRV") == 0) {
    float v[3];
    if (!parseFloats(rest, v, 3)) goto bad;
    // Strafe is +right in the protocol: D is +vy, A is -vy. Rotate: → is +w.
    pressed(drv[0], v[0], 1, 2, SHORT_ON, SHORT_OFF, F("W/S"));
    pressed(drv[1], v[1], 4, 3, SHORT_ON, SHORT_OFF, F("D/A"));
    pressed(drv[2], v[2], 6, 5, SHORT_ON, SHORT_OFF, F("right/left"));
    memcpy(drv, v, sizeof drv);
  } else if (strcmp(line, "GIM") == 0) {
    float v[2];
    if (!parseFloats(rest, v, 2)) goto bad;
    pressed(gimRate[0], v[0], 1, 2, LONG_ON, LONG_OFF, F("I/K"));
    pressed(gimRate[1], v[1], 4, 3, LONG_ON, LONG_OFF, F("L/J"));
    memcpy(gimRate, v, sizeof gimRate);
    pitchDeg = constrain(pitchDeg + gimRate[0] * GIM_DEG_PER_TICK, -90.0, 90.0);
    yawDeg = fmod(yawDeg + gimRate[1] * GIM_DEG_PER_TICK + 360.0, 360.0);
  } else if (strcmp(line, "FIRE") == 0) {
    long n = strtol(rest, NULL, 10);
    if (n < 1 || n > 10) goto bad;
    Serial.print(F("FIRE "));
    Serial.println(n);
    play(10, 50, 50);
  } else if (strcmp(line, "INTAKE") == 0) {
    if (strcmp(rest, "0") != 0 && strcmp(rest, "1") != 0) goto bad;
    intakeOn = rest[0] == '1';
    Serial.print(F("INTAKE "));
    Serial.println(intakeOn ? 1 : 0);
    play(1, 1500, 0);
  } else {
    goto bad;
  }
  return;

bad:
  Serial.print(F("bad line: "));
  Serial.print(line);
  if (*rest) {
    Serial.print(' ');
    Serial.print(rest);
  }
  Serial.println();
}

void sendStatus() {
  // flags: bit 0 armed (not known here), bit 1 intake, bit 2 fault
  Serial1.print(F("STA 0.00 "));
  Serial1.print(pitchDeg, 1);
  Serial1.print(' ');
  Serial1.print(yawDeg, 1);
  Serial1.print(' ');
  Serial1.println(intakeOn ? 2 : 0);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(DEBUG_BAUD);
  Serial1.begin(LINK_BAUD);
  play(3, 80, 80);            // boot flashes, so a reset is visible without the monitor
  Serial.println(F("mega_station_test ready, listening on Serial1"));
}

// Feeds one byte; runs handleLine on each complete line.
void feed(LineReader &r, char c) {
  if (c == '\r') return;
  if (c == '\n') {
    if (r.overflow) {
      Serial.println(F("bad line: too long, dropped"));
    } else if (r.len > 0) {
      r.buf[r.len] = '\0';
      handleLine(r.buf);
    }
    r.len = 0;
    r.overflow = false;
  } else if (r.len < MAX_LINE_LEN) {
    r.buf[r.len++] = c;
  } else {
    r.overflow = true;
  }
}

void loop() {
  while (Serial1.available() > 0) {
    lastRxMs = millis();
    feed(linkReader, Serial1.read());
  }
  while (Serial.available() > 0) {
    lastRxMs = millis();
    feed(usbReader, Serial.read());
  }

  unsigned long now = millis();
  if (!stopped && now - lastRxMs > TIMEOUT_MS) {
    allStop(F("timeout, nothing on Serial1"));
  }
  if (havePing && now - lastPingMs <= TIMEOUT_MS && (long)(now - nextStaMs) >= 0) {
    sendStatus();
    nextStaMs = now + STA_EVERY_MS;
  }
  updateLed();
}
