"""
sensors.py
Датчики: ультразвуковий HC‑SR04 та геркон.
"""

import time
import RPi.GPIO as GPIO
import config

GPIO.setmode(GPIO.BCM)
GPIO.setup(config.ULTRASONIC_TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(config.ULTRASONIC_ECHO_PIN, GPIO.IN)
GPIO.setup(config.REED_SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

SOUND_SPEED_CM_S = 34300
DISTANCE_THRESHOLD_CM = 50

def _pulse_trigger():
    GPIO.output(config.ULTRASONIC_TRIGGER_PIN, GPIO.HIGH)
    time.sleep(1e-5)
    GPIO.output(config.ULTRASONIC_TRIGGER_PIN, GPIO.LOW)

    start = time.time()
    timeout = start + 0.04
    while GPIO.input(config.ULTRASONIC_ECHO_PIN) == 0 and time.time() < timeout:
        start = time.time()
    stop = time.time()
    timeout = stop + 0.04
    while GPIO.input(config.ULTRASONIC_ECHO_PIN) == 1 and time.time() < timeout:
        stop = time.time()
    return stop - start

def read_distance() -> float:
    elapsed = _pulse_trigger()
    return (elapsed * SOUND_SPEED_CM_S) / 2

def is_car_passing(threshold: float = DISTANCE_THRESHOLD_CM) -> bool:
    return read_distance() < threshold

def is_gate_closed() -> bool:
    return GPIO.input(config.REED_SWITCH_PIN) == GPIO.HIGH

def is_gate_open() -> bool:
    return GPIO.input(config.REED_SWITCH_PIN) == GPIO.LOW

def cleanup() -> None:
    GPIO.cleanup()
