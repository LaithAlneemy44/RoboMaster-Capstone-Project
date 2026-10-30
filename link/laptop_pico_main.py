# Laptop-side Pico 2 WH: joins the robot Pico's access point, reads
# newline-terminated commands from the MacBook over USB serial and sends
# each one as a UDP packet. Datagrams coming back (telemetry from the robot)
# are written to USB serial for the MacBook. Save to the Pico as main.py.

import network
import socket
import select
import sys
import time
from link_secrets import PASSWORD   # gitignored; see link_secrets.example.py
from machine import Pin

SSID = "robo-link"
ROBOT = ("192.168.4.1", 4210)

led = Pin("LED", Pin.OUT)
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
try:
    wlan.config(pm=network.WLAN.PM_NONE)
except Exception:
    pass


JOIN_TIMEOUT_MS = 10000      # give up on one attempt after this, then retry


def connect():
    # Retry until joined. Each attempt drops any stale association first: a bare
    # wlan.connect() never rejoins after ROBOT restarts its AP, and after a soft
    # reset isconnected() can report True for a link that is already dead.
    led.off()
    while True:
        wlan.disconnect()
        wlan.connect(SSID, PASSWORD)
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < JOIN_TIMEOUT_MS:
            if wlan.isconnected():
                led.on()
                return
            time.sleep_ms(100)


def open_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    return s


connect()
sock = open_socket()

poller = select.poll()
poller.register(sys.stdin, select.POLLIN)

while True:
    if not wlan.isconnected():
        connect()
        sock.close()
        sock = open_socket()
    if poller.poll(10):
        line = sys.stdin.readline()
        if line:
            try:
                sock.sendto(line.encode(), ROBOT)
            except OSError as e:
                # Was silently dropped before 2026-10-30, which hid a relay that stopped
                # forwarding after both Picos powered up together. Say so, and start over
                # with a fresh socket and association. The host ignores this line: it
                # parses only STA/ACK/ERR.
                print("send failed:", e)
                sock.close()
                connect()
                sock = open_socket()
    try:
        reply = sock.recv(128)
    except OSError:
        reply = None
    if reply:
        sys.stdout.write(reply.decode())
