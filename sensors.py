"""
GPIO-утиліти:
* керування реле (LOW-level, активні нулем)
* замір відстані датчиком AJ-SR04M (HC-SR04-сумісний)
* читання геркона (нормально-розімкнений, pull-up)
"""

import time
import contextlib

import RPi.GPIO as GPIO

import config

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ─── Ініціалізація пінів ───────────────────────────────────────────────────────
GPIO.setup(config.RELAY_OPEN_PIN,  GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(config.RELAY_CLOSE_PIN, GPIO.OUT, initial=GPIO.HIGH)

GPIO.setup(config.ULTRASONIC_TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(config.ULTRASONIC_ECHO_PIN,    GPIO.IN)

GPIO.setup(config.REED_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# ─── Реле ──────────────────────────────────────────────────────────────────────
def relay_open(pulse=0.5):
    GPIO.output(config.RELAY_OPEN_PIN, GPIO.LOW)
    time.sleep(pulse)
    GPIO.output(config.RELAY_OPEN_PIN, GPIO.HIGH)


def relay_close(pulse=0.5):
    GPIO.output(config.RELAY_CLOSE_PIN, GPIO.LOW)
    time.sleep(pulse)
    GPIO.output(config.RELAY_CLOSE_PIN, GPIO.HIGH)


# ─── Ультразвуковий датчик ─────────────────────────────────────────────────────
SPEED_SOUND = 343.0  # м/с

def get_distance_cm() -> float:
    """
    Повертає відстань у сантиметрах або inf, якщо немає еха (таймаут 40 мс).
    """
    # одиночний імпульс 10 мкс
    GPIO.output(config.ULTRASONIC_TRIGGER_PIN, GPIO.HIGH)
    time.sleep(10e-6)
    GPIO.output(config.ULTRASONIC_TRIGGER_PIN, GPIO.LOW)

    # чекаємо фронт
    t0 = time.time()
    while GPIO.input(config.ULTRASONIC_ECHO_PIN) == 0:
        if time.time() - t0 > 0.04:
            return float("inf")

    start = time.time()
    while GPIO.input(config.ULTRASONIC_ECHO_PIN) == 1:
        if time.time() - start > 0.04:
            return float("inf")

    stop = time.time()
    duration = stop - start
    return (duration * SPEED_SOUND / 2) * 100  # двосторонній шлях


# ─── Геркон ────────────────────────────────────────────────────────────────────
def gate_is_closed() -> bool:
    """True, якщо геркон замкнений (ворота в закритому положенні)."""
    return GPIO.input(config.REED_PIN) == GPIO.LOW


# ─── Прибирання ────────────────────────────────────────────────────────────────
def cleanup():
    with contextlib.suppress(Exception):
        GPIO.cleanup()
