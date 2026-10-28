// Arduino MKR Zero stand-in for the robot, for running the real driver-station GUI
// (scripts/robot_gui.py --driver serial) over the Pico WiFi link with nothing that moves.
//
// Understands the GUI's protocol (scripts/robot/protocol.py, docs/ROBOT_PROTOCOL.md),
// shows it on the built-in LED, and answers with STA telemetry so the GUI sees a live link.
// Every command that changes something is echoed to the USB Serial monitor.
//
//   DRV vx vy w      chassis demand, -1..1      LED on while any drive or turret demand != 0
//   GIM pitch yaw    turret RATE, -1..1         integrated into the reported pitch/yaw
//   FIRE n           launch n                   n quick flashes
//   INTAKE 0|1       collector                  reported in the STA flags
//   STOP / S         all stop (S is the ROBOT Pico's own failsafe)
//   PING seq         heartbeat
//
// Telemetry: "STA <battery> <pitch> <yaw> <flags>" at 10 Hz on Serial1 TX, but only while
// PINGs keep arriving. Answering regardless would show "link: OK" in the GUI with the
// downlink dead; this way OK means the round trip works. There is no battery sensor, so
// battery is always reported as 0.
//
// Wiring: Pico GP0 (pin 1) -> MKR pin 13 (RX), MKR pin 14 (TX) -> Pico GP1 (pin 2),
// GND -> GND. Both boards are 3.3 V, so no divider. (A Mega's 5 V TX would need one.)

#include <Arduino.h>

const unsigned long LINK_BAUD   = 115200;
const unsigned long DEBUG_BAUD  = 115200;
const unsigned long TIMEOUT_MS  = 300;    // protocol.TIMEOUT_MS: stop after this much silence
const unsigned long STA_EVERY_MS = 100;   // 10 Hz, matching the GUI's heartbeat
const float GIM_DEG_PER_TICK = 2.0;       // same integration as driver.MockDriver

const uint8_t MAX_LINE_LEN = 64;
char lineBuf[MAX_LINE_LEN + 1];
uint8_t lineLen = 0;
bool lineOverflow = false;

float drv[3] = {0, 0, 0};
float gimRate[2] = {0, 0};
float pitchDeg = 0, yawDeg = 0;
bool intakeOn = false;
bool moving = false;

unsigned long lastRxMs = 0;
unsigned long lastPingMs = 0;
bool havePing = false;
bool stopped = true;
unsigned long nextStaMs = 0;

// FIRE flashes override the motion indicator until they finish.
uint8_t flashPhasesLeft = 0;
unsigned long nextFlashMs = 0;

// After a software reset macOS misses the microsecond USB drop and loses the port;
// detaching for long enough forces a clean re-enumeration. See link/README.md.
void reenumerateAfterSoftReset() {
  if (PM->RCAUSE.bit.POR) return;
  USBDevice.detach();
  delay(500);
  USBDevice.attach();
}

void updateLed() {
  if (flashPhasesLeft > 0) {
    if ((long)(millis() - nextFlashMs) >= 0) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      flashPhasesLeft--;
      nextFlashMs += 60;
    }
    return;
  }
  digitalWrite(LED_BUILTIN, moving ? HIGH : LOW);
}

void allStop(const __FlashStringHelper *why) {
  drv[0] = drv[1] = drv[2] = 0;
  gimRate[0] = gimRate[1] = 0;
  intakeOn = false;
  moving = false;
  if (!stopped) {
    Serial.print(F("STOP ("));
    Serial.print(why);
    Serial.println(F(")"));
  }
  stopped = true;
}

// Parses up to `n` floats after the command word into `out`. False unless exactly `n`.
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

void printVec(const __FlashStringHelper *name, const float *v, uint8_t n) {
  Serial.print(name);
  for (uint8_t i = 0; i < n; i++) {
    Serial.print(' ');
    Serial.print(v[i], 3);
  }
  Serial.println();
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
    if (v[0] != drv[0] || v[1] != drv[1] || v[2] != drv[2]) {
      memcpy(drv, v, sizeof drv);
      printVec(F("DRV"), drv, 3);             // only on change: it arrives at 30 Hz
    }
  } else if (strcmp(line, "GIM") == 0) {
    float v[2];
    if (!parseFloats(rest, v, 2)) goto bad;
    if (v[0] != gimRate[0] || v[1] != gimRate[1]) {
      memcpy(gimRate, v, sizeof gimRate);
      printVec(F("GIM"), gimRate, 2);
    }
    pitchDeg = constrain(pitchDeg + gimRate[0] * GIM_DEG_PER_TICK, -90.0, 90.0);
    yawDeg = fmod(yawDeg + gimRate[1] * GIM_DEG_PER_TICK + 360.0, 360.0);
  } else if (strcmp(line, "FIRE") == 0) {
    long n = strtol(rest, NULL, 10);
    if (n < 1 || n > 10) goto bad;
    Serial.print(F("FIRE "));
    Serial.println(n);
    flashPhasesLeft = 6;                      // three quick flashes
    nextFlashMs = millis();
  } else if (strcmp(line, "INTAKE") == 0) {
    if (strcmp(rest, "0") != 0 && strcmp(rest, "1") != 0) goto bad;
    intakeOn = rest[0] == '1';
    Serial.print(F("INTAKE "));
    Serial.println(intakeOn ? 1 : 0);
  } else {
    goto bad;
  }
  moving = drv[0] != 0 || drv[1] != 0 || drv[2] != 0 || gimRate[0] != 0 || gimRate[1] != 0;
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
  reenumerateAfterSoftReset();
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(DEBUG_BAUD);   // native USB; no while(!Serial), so it runs on a charger too
  Serial1.begin(LINK_BAUD);
  flashPhasesLeft = 6;        // boot flash, so a reset is visible without the monitor
  nextFlashMs = millis();
  Serial.println(F("mkr_station_test ready, listening on Serial1"));
}

void loop() {
  while (Serial1.available() > 0) {
    char c = Serial1.read();
    lastRxMs = millis();
    if (c == '\r') continue;
    if (c == '\n') {
      if (lineOverflow) {
        Serial.println(F("bad line: too long, dropped"));
      } else if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < MAX_LINE_LEN) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;
    }
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
