"""Read the board's serial console for N seconds and print what arrives.

Usage: serial-listen.py <device> <seconds> [send-enter]
"""
import os
import sys
import termios
import time

dev, secs = sys.argv[1], float(sys.argv[2])
fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
attrs = termios.tcgetattr(fd)
attrs[4] = attrs[5] = termios.B115200
attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
attrs[0] = attrs[1] = attrs[3] = 0
termios.tcsetattr(fd, termios.TCSANOW, attrs)
if len(sys.argv) > 3:
    os.write(fd, b"\r\n")
end = time.time() + secs
buf = b""
while time.time() < end:
    try:
        chunk = os.read(fd, 4096)
        if chunk:
            buf += chunk
    except BlockingIOError:
        time.sleep(0.05)
os.close(fd)
text = buf.decode(errors="replace")
print(text[-6000:] if text else "(no serial output)")
