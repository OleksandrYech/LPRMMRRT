"""
config.py
Конфігураційні константи для системи керування воротами, адаптованої
під Raspberry Pi OS 64‑bit (ARM aarch64) та Raspberry Pi 5.
Змінюйте лише за потребою через змінні оточення або редагуючи файл.
"""

import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent

# ==== Google Sheets ==== #
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID', '1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4')
GOOGLE_SHEET_RANGE_ALLOWED = os.getenv('GOOGLE_SHEET_RANGE_ALLOWED', 'Vehicles!A3:E')
GOOGLE_SHEET_RANGE_UNAUTHORIZED = os.getenv('GOOGLE_SHEET_RANGE_UNAUTHORIZED', 'Vehicles!G3:K')
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', str(BASE_DIR / 'credentials.json'))

# ==== GPIO під'єднання ==== #
# Реле
RELAY_OPEN_PIN  = 17
RELAY_CLOSE_PIN = 27
RELAY_PULSE_DURATION = float(os.getenv('RELAY_PULSE_DURATION', '0.5'))  # сек

# Ультразвук
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN    = 24

# Геркон
REED_SWITCH_PIN = 22

# ==== Камери ==== #
ENTRY_CAMERA_INDEX = int(os.getenv('ENTRY_CAMERA_INDEX', '0'))
EXIT_CAMERA_INDEX  = int(os.getenv('EXIT_CAMERA_INDEX',  '1'))

# ==== UltimateALPR SDK ==== #
ALPR_ASSETS_DIR = pathlib.Path(os.getenv(
    'ALPR_ASSETS_DIR',
    str(BASE_DIR.parent / 'ultimateALPR-SDK' / 'assets')
)).as_posix()

ALPR_JSON_CONFIG = {
    "debug_level": "info",
    "gpgpu_enabled": True,
    "num_threads": -1,
    "max_latency": -1,
    "charset": "latin",
    "assets_folder": ALPR_ASSETS_DIR,
    "klass_vcr_enabled": True,
    "klass_vmmr_enabled": True
}
