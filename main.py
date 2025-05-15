"""
Головний скрипт роботи системи. Запускає:
* MONITOR — відстежує машину біля в’їзду (ультразвук + ALPR)
* EXIT    — обробляє кнопку/команду виїзду
"""

import signal
import sys
import threading
import time

import cv2

import sensors
from detection import recognize_bgr


CAMERA_ID   = 0         # /dev/video0
DIST_LIMIT  = 120       # см, ближче — вважаємо «є авто»
PLATE_SCORE = 0.75      # мінімальна впевненість ALPR


def entry_worker():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        if sensors.get_distance_cm() > DIST_LIMIT:
            time.sleep(0.1)
            continue

        ok, frame = cap.read()
        if not ok:
            continue

        result = recognize_bgr(frame)
        if not result:
            continue

        plate = result["plates"][0]
        if plate["score"] < PLATE_SCORE:
            continue

        text = plate["text"]
        print(f"[ENTRY] Detected: {text}")
        sensors.relay_open()
        # TODO: запис у Google Sheets


def exit_worker():
    """
    Спрощений сценарій: відкривати при натисканні 'e' у консолі.
    Можна замінити на власний сенсор / модуль.
    """
    while True:
        ch = sys.stdin.read(1)
        if ch.lower() == "e":
            sensors.relay_open()
            print("[EXIT] Manual open requested")


def main():
    t_entry = threading.Thread(target=entry_worker, daemon=True)
    t_exit  = threading.Thread(target=exit_worker,  daemon=True)
    t_entry.start()
    t_exit.start()

    print("System up. Press Ctrl-C to stop.")
    signal.pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        sensors.cleanup()
