// Scripted demo routine for the Arduino Mega 2560 - no vision, no laptop.
// Also builds for an Arduino Uno (Jude's drive test board): the pin set is chosen at compile
// time from the board selected in the IDE.
//
// Powers up, waits STARTUP_DELAY_MS, runs ROUTINE once, then stops everything and idles.
// Power-cycle to replay. There are no buttons: power off is the stop.
//
// Hardware (from the procurement sheet):
//   Drive     - 4x DFRobot FIT0186 12 V on 2x Cytron MDD10A (PWM + DIR per motor), mecanum
//   Collector - geared DC motor on a TB6612FNG channel A, one direction
//   Shooter   - yellow TT motor switched low-side by an IRLB8721 MOSFET, run for a fixed time
// Not used yet: the two MG995 aiming servos, the MPU6050 and the shooter limit switch - to be
// merged from Binada's shooter code.
//
// Wiring notes:
//   - A ball is loaded by hand before power-up; the routine shoots it, then runs the collector.
//   - The MG995 servos take power from the 5 V buck, never the Mega's 5 V pin (they can draw
//     over 1 A each at stall and would reset the board). Only the signal wire goes to the
//     Mega; all grounds are common. Check the buck's current rating covers both servos.
//   - Emergency stop is unplugging the supply: keep that connector reachable during the demo.

#include <Arduino.h>  // the Arduino build adds this itself; explicit for editor tooling

// ---- Subsystems: switch off what isn't connected (Jude's Uno test is drive only) ----
const bool ENABLE_COLLECTOR = false;
const bool ENABLE_SHOOTER   = false;

// ---- Pins (TODO confirm every one against the real wiring; 0/1 left free for upload) ----
#if defined(__AVR_ATmega2560__)
const char BOARD_NAME[] = "Mega";
// Drive PWM avoids 2, 3 and 18-21: those are the interrupt pins the encoders would need.
const uint8_t FL_PWM = 4,  FL_DIR = 22;   // MDD10A #1, channel 1
const uint8_t FR_PWM = 5,  FR_DIR = 23;   // MDD10A #1, channel 2
const uint8_t RL_PWM = 6,  RL_DIR = 24;   // MDD10A #2, channel 1
const uint8_t RR_PWM = 7,  RR_DIR = 25;   // MDD10A #2, channel 2
const uint8_t COL_PWMA = 8, COL_AIN1 = 26, COL_AIN2 = 27, COL_STBY = 28;  // TB6612FNG
const uint8_t SHOOT_GATE = 9;             // IRLB8721 gate
#else
const char BOARD_NAME[] = "Uno";
// The Uno has exactly six PWM pins (3, 5, 6, 9, 10, 11) - one per motor here.
const uint8_t FL_PWM = 3,  FL_DIR = 2;    // MDD10A #1, channel 1
const uint8_t FR_PWM = 5,  FR_DIR = 4;    // MDD10A #1, channel 2
const uint8_t RL_PWM = 6,  RL_DIR = 7;    // MDD10A #2, channel 1
const uint8_t RR_PWM = 9,  RR_DIR = 8;    // MDD10A #2, channel 2
const uint8_t COL_PWMA = 10, COL_AIN1 = 12, COL_AIN2 = 13, COL_STBY = A0;  // TB6612FNG
const uint8_t SHOOT_GATE = 11;            // IRLB8721 gate
#endif

// ---- Speeds (0-255) ----
const uint8_t DRIVE_SPEED   = 150;
const uint8_t COLLECT_SPEED = 180;        // below full until the motor's stall current is known
const uint8_t SHOOT_SPEED   = 255;        // 255 = MOSFET fully on

// ---- Direction flags: flip one if that motor runs backwards, no rewiring needed ----
const bool INVERT_FL = false, INVERT_FR = false, INVERT_RL = false, INVERT_RR = false;
const bool INVERT_COLLECTOR = false;

// ---- Timing ----
const unsigned long STARTUP_DELAY_MS = 3000;  // time to step back after power-on
const unsigned long RAMP_MS          = 200;   // drive PWM ramp, avoids brown-out resets
const uint16_t      PAUSE_MS         = 700;

// ---- Routine: tune durations here ----
enum Action { FORWARD, BACKWARD, ROT_LEFT, ROT_RIGHT, SHOOT, COLLECT, PAUSE };
const char* const ACTION_NAMES[] = {
  "FORWARD", "BACKWARD", "ROT_LEFT", "ROT_RIGHT", "SHOOT", "COLLECT", "PAUSE"
};

struct Step { Action action; uint16_t ms; };

const Step ROUTINE[] = {
  {FORWARD,   1000}, {PAUSE, PAUSE_MS},
  {BACKWARD,  1000}, {PAUSE, PAUSE_MS},
  {ROT_LEFT,   800}, {PAUSE, PAUSE_MS},
  {ROT_RIGHT,  800}, {PAUSE, PAUSE_MS},
  {SHOOT,     1500}, {PAUSE, PAUSE_MS},
  {COLLECT,   2000},
};
const uint8_t STEP_COUNT = sizeof(ROUTINE) / sizeof(ROUTINE[0]);

// ---- State ----
int stepIndex = -1;              // -1 = startup delay, STEP_COUNT = finished
unsigned long stepStart = 0;

// ---- Hardware helpers ----
void setDrive(uint8_t pwmPin, uint8_t dirPin, int speed, bool invert) {
  if (invert) speed = -speed;
  digitalWrite(dirPin, speed >= 0 ? HIGH : LOW);
  analogWrite(pwmPin, abs(speed));
}

// Each argument is -1, 0 or +1 for that wheel; pwm is the ramped magnitude.
void driveWheels(int fl, int fr, int rl, int rr, int pwm) {
  setDrive(FL_PWM, FL_DIR, fl * pwm, INVERT_FL);
  setDrive(FR_PWM, FR_DIR, fr * pwm, INVERT_FR);
  setDrive(RL_PWM, RL_DIR, rl * pwm, INVERT_RL);
  setDrive(RR_PWM, RR_DIR, rr * pwm, INVERT_RR);
}

void setCollector(uint8_t speed) {
  if (!ENABLE_COLLECTOR) return;
  digitalWrite(COL_STBY, speed > 0 ? HIGH : LOW);
  digitalWrite(COL_AIN1, INVERT_COLLECTOR ? LOW : HIGH);
  digitalWrite(COL_AIN2, INVERT_COLLECTOR ? HIGH : LOW);
  analogWrite(COL_PWMA, speed);
}

void setShooter(uint8_t speed) {
  if (!ENABLE_SHOOTER) return;
  analogWrite(SHOOT_GATE, speed);
}

void allStop() {
  driveWheels(0, 0, 0, 0, 0);
  setCollector(0);
  setShooter(0);
}

// ---- Apply the current step; called every loop so the drive ramp updates ----
void runStep(Action action, unsigned long elapsed) {
  int pwm = DRIVE_SPEED * min(elapsed, RAMP_MS) / RAMP_MS;
  switch (action) {
    case FORWARD:   driveWheels( 1,  1,  1,  1, pwm); break;
    case BACKWARD:  driveWheels(-1, -1, -1, -1, pwm); break;
    case ROT_LEFT:  driveWheels(-1,  1, -1,  1, pwm); break;
    case ROT_RIGHT: driveWheels( 1, -1,  1, -1, pwm); break;
    case SHOOT:     setShooter(SHOOT_SPEED);           break;
    case COLLECT:   setCollector(COLLECT_SPEED);       break;
    case PAUSE:     allStop();                         break;
  }
}

void setup() {
  const uint8_t drivePins[] = {FL_PWM, FL_DIR, FR_PWM, FR_DIR, RL_PWM, RL_DIR, RR_PWM, RR_DIR};
  for (uint8_t pin : drivePins) pinMode(pin, OUTPUT);
  if (ENABLE_COLLECTOR) {
    const uint8_t collectorPins[] = {COL_PWMA, COL_AIN1, COL_AIN2, COL_STBY};
    for (uint8_t pin : collectorPins) pinMode(pin, OUTPUT);
  }
  if (ENABLE_SHOOTER) pinMode(SHOOT_GATE, OUTPUT);
  allStop();

  Serial.begin(115200);
  Serial.print(F("board: "));
  Serial.print(BOARD_NAME);
  Serial.print(ENABLE_COLLECTOR ? F(", collector on") : F(", collector off"));
  Serial.println(ENABLE_SHOOTER ? F(", shooter on") : F(", shooter off"));
  Serial.print(F("demo_routine: starting in "));
  Serial.print(STARTUP_DELAY_MS);
  Serial.println(F(" ms"));
  stepStart = millis();
}

void loop() {
  if (stepIndex >= STEP_COUNT) return;   // finished; idle until power-cycled

  unsigned long now = millis();
  unsigned long elapsed = now - stepStart;
  unsigned long duration = stepIndex < 0 ? STARTUP_DELAY_MS : ROUTINE[stepIndex].ms;

  if (elapsed >= duration) {
    allStop();                           // every step starts from rest
    stepIndex++;
    stepStart = now;
    elapsed = 0;
    if (stepIndex >= STEP_COUNT) {
      Serial.println(F("done - power-cycle to replay"));
      return;
    }
    Serial.print(F("step "));
    Serial.print(stepIndex);
    Serial.print(F(": "));
    Serial.print(ACTION_NAMES[ROUTINE[stepIndex].action]);
    Serial.print(F(" "));
    Serial.print(ROUTINE[stepIndex].ms);
    Serial.print(F(" ms"));
    Action a = ROUTINE[stepIndex].action;
    bool disabled = (a == SHOOT && !ENABLE_SHOOTER) || (a == COLLECT && !ENABLE_COLLECTOR);
    Serial.println(disabled ? F(" (disabled)") : F(""));
  }

  if (stepIndex >= 0) runStep(ROUTINE[stepIndex].action, elapsed);
}
