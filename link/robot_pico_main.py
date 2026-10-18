# Robot-side Pico 2 WH: hosts the WiFi access point, receives UDP
# commands from the laptop-side Pico and forwards them to the Arduino Mega
# over UART. Save to the Pico as main.py so it runs on power-up.

import network
import socket
import time
from link_secrets import PASSWORD   # gitignored; see link_secrets.example.py
from machine import UART, Pin

SSID = "robo-link"
PORT = 4210
FAILSAFE_MS = 300             # no packet for this long -> send stop
STOP_CMD = b"S\n"

led = Pin("LED", Pin.OUT)

# --- WiFi access point ---
ap = network.WLAN(network.AP_IF)
ap.config(essid=SSID, password=PASSWORD)
ap.active(True)
while not ap.active():
    time.sleep_ms(50)
try:
    ap.config(pm=network.WLAN.PM_NONE)   # disable power saving for latency
except Exception:
    pass
print("AP up:", ap.ifconfig())           # default IP is 192.168.4.1
led.on()                                 # power/AP indicator; toggles per packet from here

# --- UART to Arduino Mega: GP0 = TX, GP1 = RX ---
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# --- UDP socket ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.setblocking(False)

last_rx = time.ticks_ms()
stopped = True

while True:
    try:
        data = sock.recv(64)
    except OSError:
        data = None

    now = time.ticks_ms()
    if data:
        uart.write(data)
        print("->", data)                # visible over USB when testing on the Mac
        led.toggle()
        last_rx = now
        stopped = False
    elif not stopped and time.ticks_diff(now, last_rx) > FAILSAFE_MS:
        uart.write(STOP_CMD)
        print("failsafe stop")
        stopped = True
