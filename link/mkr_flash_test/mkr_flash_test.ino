// Arduino MKR Zero stand-in for the Mega, for testing the Pico WiFi link without the robot.
//
// Reads newline-terminated lines from the ROBOT Pico on Serial1 and plays an LED pattern for
// each flash command, so a key press in link/flash_test_gui.py shows up as a visible blink.
// Every line received is echoed to the USB Serial monitor. Nothing here blocks.
//
// Commands:
//   L,<seq>,<code>   flash command. code: 1-5 = that many blinks, F = fast flicker,
//                    H = hold on for 1 s, 0 = LED off
//   S                stop - the ROBOT Pico sends this 300 ms after every burst of packets.
//                    Logged only: it would otherwise cut every pattern short.
//
// Wiring: Pico GP0 (pin 1, TX) -> MKR pin 13 (RX), Pico GND (pin 3) -> MKR GND.
// Both boards are 3.3 V, so no level shifting is needed in either direction.

#include <Arduino.h>

const unsigned long LINK_BAUD  = 115200;  // must match the ROBOT Pico's UART
const unsigned long DEBUG_BAUD = 115200;

const uint8_t MAX_LINE_LEN = 64;
char lineBuf[MAX_LINE_LEN + 1];
uint8_t lineLen = 0;
bool lineOverflow = false;

long lastSeq = -1;

// Pattern player: the LED starts on and toggles every phaseMs until phasesLeft runs out.
bool patternActive = false;
uint8_t phasesLeft = 0;
unsigned long phaseMs = 0;
unsigned long nextToggleMs = 0;

void ledOff() {
  patternActive = false;
  digitalWrite(LED_BUILTIN, LOW);
}

void startPattern(uint8_t phases, unsigned long ms) {
  digitalWrite(LED_BUILTIN, HIGH);
  patternActive = true;
  phasesLeft = phases - 1;
  phaseMs = ms;
  nextToggleMs = millis() + ms;
}

void updatePattern() {
  if (!patternActive || (long)(millis() - nextToggleMs) < 0) return;
  if (phasesLeft == 0) {
    ledOff();
    return;
  }
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  phasesLeft--;
  nextToggleMs += phaseMs;
}

// Returns false if `code` is not a known pattern.
bool playCode(char code) {
  if (code >= '1' && code <= '5') {
    startPattern(2 * (code - '0') - 1, 200);   // on/off pairs, ending on
  } else if (code == 'F') {
    startPattern(19, 50);
  } else if (code == 'H') {
    startPattern(1, 1000);
  } else if (code == '0') {
    ledOff();
  } else {
    return false;
  }
  return true;
}

void checkSeq(long seq) {
  if (lastSeq >= 0 && seq > lastSeq + 1) {
    Serial.print(F("WARNING: skipped "));
    Serial.print(seq - lastSeq - 1);
    Serial.println(F(" seq"));
  } else if (lastSeq >= 0 && seq <= lastSeq) {
    Serial.println(F("NOTE: seq went back, sender restarted?"));
  }
  lastSeq = seq;
}

void handleLine(char *line) {
  if (strcmp(line, "S") == 0) {
    Serial.println(F("S (link stop)"));
    return;
  }
  // Expect exactly "L,<seq>,<one char>"
  char *comma2 = NULL;
  if (line[0] == 'L' && line[1] == ',') comma2 = strchr(line + 2, ',');
  if (comma2 == NULL || comma2[1] == '\0' || comma2[2] != '\0') {
    Serial.print(F("bad line: "));
    Serial.println(line);
    return;
  }
  *comma2 = '\0';
  char *end;
  long seq = strtol(line + 2, &end, 10);
  char code = comma2[1];
  if (*end != '\0' || seq < 0 || !playCode(code)) {
    Serial.print(F("bad line: L,"));
    Serial.print(line + 2);
    Serial.print(',');
    Serial.println(code);
    return;
  }
  checkSeq(seq);
  Serial.print(F("L seq="));
  Serial.print(seq);
  Serial.print(F(" code="));
  Serial.println(code);
}

// After a software reset (an upload, or the bootloader handing over) the USB pull-up drops
// for only microseconds, so macOS never sees the board leave and keeps talking to the old
// device; the port then vanishes until a replug. Detaching for long enough makes the host see
// a real unplug and re-enumerate. Skipped on power-up, where the host is already fresh.
void reenumerateAfterSoftReset() {
  if (PM->RCAUSE.bit.POR) return;
  USBDevice.detach();
  delay(500);
  USBDevice.attach();
}

void setup() {
  reenumerateAfterSoftReset();
  pinMode(LED_BUILTIN, OUTPUT);
  ledOff();
  Serial.begin(DEBUG_BAUD);   // native USB; deliberately no while(!Serial), so it runs unplugged
  Serial1.begin(LINK_BAUD);
  startPattern(5, 100);       // boot blink, so a reset is visible without the monitor
  Serial.println(F("mkr_flash_test ready, listening on Serial1"));
}

void loop() {
  while (Serial1.available() > 0) {
    char c = Serial1.read();
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
  updatePattern();
}
