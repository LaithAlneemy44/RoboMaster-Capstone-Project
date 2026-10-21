// Arduino Mega 2560 receiver for the Pico WiFi link.
//
// Reads newline-terminated ASCII commands from the ROBOT Pico on Serial1 (pin 19 = RX1) and
// echoes every parsed command to the USB Serial monitor for debugging. Nothing here blocks:
// loop() only drains whatever bytes have arrived and returns.
//
// Commands (see link/README.md):
//   A,<seq>,<yaw>,<pitch>,<fire>   aim command, e.g. "A,1234,12.50,-3.20,1"
//   S                              stop - sent by the ROBOT Pico after 300 ms without packets
//
// Wiring: Pico GP0 (pin 1, TX) -> Mega pin 19 (RX1), Pico GND (pin 3) -> Mega GND.
// Serial1 is used because Serial (pins 0/1) is shared with the Mega's USB port.

#include <Arduino.h>  // the Arduino build adds this itself; explicit for editor tooling

const unsigned long LINK_BAUD  = 115200;  // must match the ROBOT Pico's UART
const unsigned long DEBUG_BAUD = 115200;  // Serial Monitor speed

// Local watchdog, in addition to the Pico's 300 ms failsafe: if the Pico itself dies or its
// wire comes loose, no "S" ever arrives, so the Mega stops on its own silence as well.
// Which hop owns the stop rule is still an open decision (CLAUDE.md, control link item 4).
const unsigned long LINK_TIMEOUT_MS = 500;

const uint8_t MAX_LINE_LEN = 64;              // longest accepted line, excluding the newline
char lineBuf[MAX_LINE_LEN + 1];
uint8_t lineLen = 0;
bool lineOverflow = false;

// Latest parsed command
long  cmdSeq   = -1;
float cmdYaw   = 0.0;   // degrees
float cmdPitch = 0.0;   // degrees
bool  cmdFire  = false;

long lastSeq = -1;                        // -1 = nothing received yet
unsigned long lastRxMs = 0;
bool stopped = true;

// ---------------------------------------------------------------------------------------
// PLACEHOLDERS - replace with the real actuator code. Each is called once per command and
// must return immediately: no delay(), no waiting on a motor to arrive.
// ---------------------------------------------------------------------------------------

// PLACEHOLDER: drive the yaw (left/right) aiming motor toward `yawDeg`.
void setYaw(float yawDeg) {
  (void)yawDeg;
  // e.g. yawServo.write(constrain(90 + yawDeg, 0, 180));
}

// PLACEHOLDER: drive the pitch (up/down) aiming motor toward `pitchDeg`.
void setPitch(float pitchDeg) {
  (void)pitchDeg;
  // e.g. pitchServo.write(constrain(90 + pitchDeg, 0, 180));
}

// PLACEHOLDER: run the firing mechanism while `fire` is true, stop it when false.
// If a shot is a timed pulse, track its start with millis() here rather than delay().
void setFire(bool fire) {
  (void)fire;
  // e.g. analogWrite(SHOOT_GATE, fire ? 255 : 0);
}

// PLACEHOLDER: stop every actuator - hold aim, shooter off. Must be safe to call repeatedly.
void stopAll() {
  setFire(false);
  // e.g. hold the servos where they are, cut drive motors
}

// ---------------------------------------------------------------------------------------

void handleStop(const __FlashStringHelper *why) {
  stopAll();
  stopped = true;
  digitalWrite(LED_BUILTIN, LOW);
  Serial.print(F("STOP ("));
  Serial.print(why);
  Serial.println(F(")"));
}

// Parses "A,seq,yaw,pitch,fire" in place. Returns false on any malformed field.
bool parseAim(char *s) {
  char *fields[5];
  uint8_t n = 0;
  for (char *tok = strtok(s, ","); tok != NULL; tok = strtok(NULL, ",")) {
    if (n == 5) return false;               // too many fields
    fields[n++] = tok;
  }
  if (n != 5) return false;
  if (strcmp(fields[0], "A") != 0) return false;

  char *end;
  long seq = strtol(fields[1], &end, 10);
  if (*end != '\0' || seq < 0) return false;
  double yaw = strtod(fields[2], &end);
  if (*end != '\0') return false;
  double pitch = strtod(fields[3], &end);
  if (*end != '\0') return false;
  if (strcmp(fields[4], "0") != 0 && strcmp(fields[4], "1") != 0) return false;

  cmdSeq   = seq;
  cmdYaw   = yaw;
  cmdPitch = pitch;
  cmdFire  = fields[4][0] == '1';
  return true;
}

void checkSeq(long seq) {
  if (lastSeq >= 0) {
    if (seq > lastSeq + 1) {
      Serial.print(F("WARNING: skipped "));
      Serial.print(seq - lastSeq - 1);
      Serial.print(F(" seq ("));
      Serial.print(lastSeq);
      Serial.print(F(" -> "));
      Serial.print(seq);
      Serial.println(F(")"));
    } else if (seq <= lastSeq) {
      // A lower number usually means the sender restarted, not a lost packet.
      Serial.print(F("NOTE: seq went back ("));
      Serial.print(lastSeq);
      Serial.print(F(" -> "));
      Serial.print(seq);
      Serial.println(F("), sender restarted?"));
    }
  }
  lastSeq = seq;
}

void handleLine(char *line) {
  if (strcmp(line, "S") == 0) {
    handleStop(F("link"));
    return;
  }
  char copy[MAX_LINE_LEN + 1];
  strcpy(copy, line);       // parseAim cuts `line` up at the commas
  if (!parseAim(line)) {
    Serial.print(F("bad line: "));
    Serial.println(copy);
    return;
  }
  checkSeq(cmdSeq);
  stopped = false;

  setYaw(cmdYaw);
  setPitch(cmdPitch);
  setFire(cmdFire);
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));  // toggles per command, like the ROBOT Pico

  Serial.print(F("A seq="));
  Serial.print(cmdSeq);
  Serial.print(F(" yaw="));
  Serial.print(cmdYaw, 2);
  Serial.print(F(" pitch="));
  Serial.print(cmdPitch, 2);
  Serial.print(F(" fire="));
  Serial.println(cmdFire ? 1 : 0);
}

void setup() {
  Serial.begin(DEBUG_BAUD);
  Serial1.begin(LINK_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);   // pin 13: flickers while commands arrive, off when stopped
  digitalWrite(LED_BUILTIN, LOW);
  stopAll();
  Serial.println(F("mega_receiver ready, listening on Serial1"));
}

void loop() {
  while (Serial1.available() > 0) {
    char c = Serial1.read();
    lastRxMs = millis();              // any byte counts as the link being alive

    if (c == '\r') continue;          // tolerate CRLF senders
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
      lineOverflow = true;            // discard the rest of this line
    }
  }

  if (!stopped && millis() - lastRxMs > LINK_TIMEOUT_MS) {
    handleStop(F("Mega timeout, nothing on Serial1"));
  }
}
