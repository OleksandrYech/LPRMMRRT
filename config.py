"""
Константи та змінні, що залежать від середовища.
Редагуйте лише за потреби — наприклад, якщо змінюєте пін-аут чи ID таблиці.
"""

import os

# ─── GPIO ──────────────────────────────────────────────────────────────────────
RELAY_OPEN_PIN        = 17
RELAY_CLOSE_PIN       = 27
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN    = 24
REED_PIN               = 22     # геркон

# ─── ALPR SDK ──────────────────────────────────────────────────────────────────
ALPR_SDK_LIB_PATH = os.getenv(
    "ALPR_SDK_LIB_PATH",
    "/usr/local/lib/libultimate_alpr-sdk.so"    # armv7l-бібліотека Doubango
)

# ─── Google Sheets ────────────────────────────────────────────────────────────
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "your-sheet-id-here")  # замініть!
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
