"""
gate.py
Керування воротами через низькорівневі реле (low‑level trigger).
"""

import time
import RPi.GPIO as GPIO
import config
from sensors import is_gate_open, is_gate_closed

GPIO.setmode(GPIO.BCM)
GPIO.setup(config.RELAY_OPEN_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(config.RELAY_CLOSE_PIN, GPIO.OUT, initial=GPIO.HIGH)

RELAY_ON  = GPIO.LOW
RELAY_OFF = GPIO.HIGH
GATE_TIMEOUT = 15

def _pulse(pin: int, checker, timeout: float) -> bool:
    GPIO.output(pin, RELAY_ON)
    time.sleep(config.RELAY_PULSE_DURATION)
    GPIO.output(pin, RELAY_OFF)
    start = time.time()
    while time.time() - start < timeout:
        if checker():
            return True
        time.sleep(0.1)
    return False

def open_gate() -> bool:
    if is_gate_open():
        return True
    return _pulse(config.RELAY_OPEN_PIN, is_gate_open, GATE_TIMEOUT)

def close_gate() -> bool:
    if is_gate_closed():
        return True
    return _pulse(config.RELAY_CLOSE_PIN, is_gate_closed, GATE_TIMEOUT)

def cleanup() -> None:
    GPIO.cleanup([config.RELAY_OPEN_PIN, config.RELAY_CLOSE_PIN])
