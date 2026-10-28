# Robot-side Pico 2 WH: hosts the WiFi access point, receives UDP
# commands from the laptop-side Pico and forwards them to the Arduino Mega
# over UART. Lines the receiver sends back on GP1 (telemetry) go to whoever
# sent the last command. Save to the Pico as main.py so it runs on power-up.

import network
import socket
import time
from link_secrets import PASSWORD   # gitignored; see link_secrets.example.py
from machine import UART, Pin

SSID = "robo-link"
PORT = 4210
FAILSAFE_MS = 300             # no packet for this long -> send stop
STOP_CMD = b"S\n"

# Self-recovery. On 2026-10-21 the AP stalled mid-test: it kept beaconing but refused
# every join (LAPTOP status -1 within 3 s) until ROBOT was power-cycled. Cause unknown,
# possibly a power dip through a jumper. Restarting the AP from here clears the same state.
NO_STATION_RESTART_MS = 20000    # nobody associated this long -> restart AP
SILENCE_RESTART_MS = 120000      # no packet this long -> restart AP (dead-but-associated link)
CHECK_MS = 1000

led = Pin("LED", Pin.OUT)

# --- WiFi access point ---
ap = network.WLAN(network.AP_IF)


def start_ap():
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


def open_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", PORT))
    s.setblocking(False)
    return s


def station_count():
    # None if this firmware can't report stations; the silence rule still applies then.
    try:
        return len(ap.status("stations"))
    except Exception:
        return None


start_ap()

# --- UART to Arduino Mega: GP0 = TX, GP1 = RX ---
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# --- UDP socket ---
sock = open_socket()

reply_to = None                  # address of the last command's sender
uart_buf = b""

last_rx = time.ticks_ms()
last_station = last_rx
last_check = last_rx
stopped = True

while True:
    try:
        data, reply_to = sock.recvfrom(64)
    except OSError:
        data = None

    # Return path: complete lines from the receiver, one per datagram. Dropped until a
    # command has arrived, since there is nobody to send them to before that.
    chunk = uart.read()
    if chunk:
        uart_buf += chunk
        while b"\n" in uart_buf:
            line, uart_buf = uart_buf.split(b"\n", 1)
            if reply_to and line:
                try:
                    sock.sendto(line + b"\n", reply_to)
                except OSError:
                    pass
        if len(uart_buf) > 256:          # no newline in sight: line noise, not a message
            uart_buf = b""

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

    if time.ticks_diff(now, last_check) >= CHECK_MS:
        last_check = now
        n = station_count()
        if n is None or n > 0:
            last_station = now
        reason = None
        if time.ticks_diff(now, last_station) > NO_STATION_RESTART_MS:
            reason = "no station associated"
        elif time.ticks_diff(now, last_rx) > SILENCE_RESTART_MS:
            reason = "no packets"
        if reason:
            print("restarting AP:", reason)
            if not stopped:                  # unreachable after 300 ms, but never skip a stop
                uart.write(STOP_CMD)
                stopped = True
            sock.close()
            ap.active(False)
            time.sleep_ms(500)
            start_ap()
            sock = open_socket()
            reply_to = None
            now = time.ticks_ms()
            last_rx = last_station = last_check = now
