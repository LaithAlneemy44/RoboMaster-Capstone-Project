
/*
  Mecanum Robot - Four Motor Test
  Controller: Mega 2560
  Drivers: 2 x Cytron MDD10A
  Motors: 4 x DFRobot FIT0186

  No encoders required.
*/

// Motor indices
const byte FL = 0;
const byte FR = 1;
const byte RL = 2;
const byte RR = 3;

// Updated pin assignments
// const byte DIR_PINS[4] = {4, 6, 11, 9};
// const byte PWM_PINS[4] = {5, 7, 10, 8};
const byte DIR_PINS[4] = {4, 7, 12, 8};
const byte PWM_PINS[4] = {5, 6, 10, 9};

// Set true where physical motor orientation
// requires the direction to be reversed.
const bool INVERT[4] = {
  false, false, false, false
};

// Test settings
// const int TEST_PWM = 77;
const int TEST_PWM = 180;
const unsigned long MOVE_TIME = 1200;
const unsigned long REST_TIME = 2000;

const int RAMP_STEPS = 20;
const int RAMP_DELAY = 20;

int currentPWM[4] = {0, 0, 0, 0};

// --------------------------------
// Control one motor
// Signed speed: -255 to +255
// --------------------------------
void setMotor(byte motor, int pwm) {
  pwm = constrain(pwm, -255, 255);

  if (INVERT[motor]) {
    pwm = -pwm;
  }

  // Disable output before changing DIR
  analogWrite(PWM_PINS[motor], 0);

  digitalWrite(DIR_PINS[motor],
               pwm >= 0 ? HIGH : LOW);

  analogWrite(PWM_PINS[motor], abs(pwm));
}

// --------------------------------
// Ramp all four motor commands
// --------------------------------
void drive(int fl, int fr, int rl, int rr) {
  int target[4] = {
    constrain(fl, -255, 255),
    constrain(fr, -255, 255),
    constrain(rl, -255, 255),
    constrain(rr, -255, 255)
  };

  int start[4];

  for (int i = 0; i < 4; i++) {
    start[i] = currentPWM[i];
  }

  for (int step = 1; step <= RAMP_STEPS; step++) {

    for (int i = 0; i < 4; i++) {
      int output = start[i] +
        (long)(target[i] - start[i]) *
        step / RAMP_STEPS;

      setMotor(i, output);
    }

    delay(RAMP_DELAY);
  }

  for (int i = 0; i < 4; i++) {
    currentPWM[i] = target[i];
  }
}

// --------------------------------
// Movement functions
// --------------------------------
void moveForward(int pwm) {
  pwm = constrain(pwm, 0, 255);
  drive(pwm, pwm, pwm, pwm);
}

void moveBackward(int pwm) {
  pwm = constrain(pwm, 0, 255);
  drive(-pwm, -pwm, -pwm, -pwm);
}

void moveRight(int pwm) {
  pwm = constrain(pwm, 0, 255);
  drive(pwm, -pwm, -pwm, pwm);
}

void moveLeft(int pwm) {
  pwm = constrain(pwm, 0, 255);
  drive(-pwm, pwm, pwm, -pwm);
}

void stopMotors() {
  drive(0, 0, 0, 0);
}

// --------------------------------
// Setup
// --------------------------------
void setup() {
  // Initialise with all PWM outputs OFF
  for (int i = 0; i < 4; i++) {
    digitalWrite(PWM_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);

    pinMode(PWM_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
  }

  Serial.begin(9600);

  Serial.println("Four Motor Mecanum Test");
  Serial.println("Initial state: OFF");

  delay(5000);

  // Forward
  Serial.println("FORWARD");
  moveForward(TEST_PWM);
  delay(MOVE_TIME);
  stopMotors();
  delay(REST_TIME);

  // Backward
  Serial.println("BACKWARD");
  moveBackward(TEST_PWM);
  delay(MOVE_TIME);
  stopMotors();
  delay(REST_TIME);

  // Right
  Serial.println("RIGHT");
  moveRight(TEST_PWM);
  delay(MOVE_TIME);
  stopMotors();
  delay(REST_TIME);

  // Left
  Serial.println("LEFT");
  moveLeft(TEST_PWM);
  delay(MOVE_TIME);
  stopMotors();

  Serial.println("TEST COMPLETE");
}

// Runs once, then stops
void loop() {
}
