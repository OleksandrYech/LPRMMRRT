import time
import RPi.GPIO as GPIO

import config
from sensors import is_gate_open, is_gate_closed

"""
gate.py

Модуль для керування реле воріт через Raspberry Pi GPIO.
Забезпечує функції безпечного відчинення та зачинення воріт за допомогою зворотного зв'язку геркона,
з можливістю налаштування тривалості імпульсу реле для забезпечення надійної активації.
"""
# GPIO setup
GPIO.setmode(GPIO.BCM)

# Relay pins
GPIO.setup(config.RELAY_OPEN_PIN, GPIO.OUT)
GPIO.setup(config.RELAY_CLOSE_PIN, GPIO.OUT)
# Ensure relays are off initially
GPIO.output(config.RELAY_OPEN_PIN, GPIO.LOW)
GPIO.output(config.RELAY_CLOSE_PIN, GPIO.LOW)

# Maximum time to wait for gate movement (seconds)
GATE_TIMEOUT = 15
# Relay activation signal level
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW

# Relay pulse duration
# Can be configured in config.py as RELAY_PULSE_DURATION (float)
RELAY_PULSE_DURATION = getattr(config, 'RELAY_PULSE_DURATION', 0.5)


def open_gate(timeout: float = GATE_TIMEOUT) -> bool:
    """
    Вмикає реле OPEN, щоб відчинити ворота.
    Забезпечує утримання реле під напругою щонайменше протягом RELAY_PULSE_DURATION секунд.
    Чекає, поки геркон не покаже, що ворота відчинено, або поки не закінчиться тайм-аут.

    param timeout: Максимальна кількість секунд для очікування відчинення воріт.
    :return: True, якщо ворота відкрито, False, якщо досягнуто тайм-ауту.
    """
    # If already open, nothing to do
    if is_gate_open():
        return True

    # Activate OPEN relay
    GPIO.output(config.RELAY_OPEN_PIN, RELAY_ON)
    start_time = time.time()

    # Wait until gate is open or timeout
    while time.time() - start_time < timeout:
        if is_gate_open():
            break
        time.sleep(0.1)

    # Ensure minimum pulse duration before deactivating
    elapsed = time.time() - start_time
    if elapsed < RELAY_PULSE_DURATION:
        time.sleep(RELAY_PULSE_DURATION - elapsed)

    # Deactivate relay
    GPIO.output(config.RELAY_OPEN_PIN, RELAY_OFF)
    return is_gate_open()


def close_gate(timeout: float = GATE_TIMEOUT) -> bool:
    """
    Вмикає реле CLOSE, щоб закрити ворота.
    Забезпечує утримання реле під напругою щонайменше протягом RELAY_PULSE_DURATION секунд.
    Чекає, поки геркон не покаже, що ворота закрито, або поки не закінчиться тайм-аут.

    param timeout: Максимальна кількість секунд для очікування закриття воріт.
    :return: True, якщо ворота закрито, False, якщо таймаут вичерпано.
    """
    # If already closed, nothing to do
    if is_gate_closed():
        return True

    # Activate CLOSE relay
    GPIO.output(config.RELAY_CLOSE_PIN, RELAY_ON)
    start_time = time.time()

    # Wait until gate is closed or timeout
    while time.time() - start_time < timeout:
        if is_gate_closed():
            break
        time.sleep(0.1)

    # Ensure minimum pulse duration before deactivating
    elapsed = time.time() - start_time
    if elapsed < RELAY_PULSE_DURATION:
        time.sleep(RELAY_PULSE_DURATION - elapsed)

    # Deactivate relay
    GPIO.output(config.RELAY_CLOSE_PIN, RELAY_OFF)
    return is_gate_closed()


def cleanup() -> None:
    """
    Очищення ресурсів GPIO для реле.
    """
    GPIO.cleanup([config.RELAY_OPEN_PIN, config.RELAY_CLOSE_PIN])