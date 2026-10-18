# Runs on the MacBook. Streams fake aim commands to the laptop-side Pico
# at 50 Hz so the full link can be tested without the vision pipeline.
# Usage: python3 mac_link_test.py /dev/cu.usbmodemXXXX

import math
import sys
import time

import serial

port = sys.argv[1]
ser = serial.Serial(port, 115200, timeout=0)

t0 = time.time()
seq = 0
try:
    while True:
        t = time.time() - t0
        yaw = 30 * math.sin(t)          # degrees, sweeps left and right
        pitch = 10 * math.sin(0.5 * t)
        fire = 1 if int(t) % 4 == 0 else 0
        ser.write(f"A,{seq},{yaw:.2f},{pitch:.2f},{fire}\n".encode())
        seq += 1
        time.sleep(0.02)
except KeyboardInterrupt:
    ser.close()
