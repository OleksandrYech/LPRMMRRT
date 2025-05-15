"""
main.py
Точка входу системи керування воротами.
"""

import threading
import time
import cv2

from detection import detect_plate_and_vehicle, detect_vehicle_exit, deinit as alpr_deinit
from sensors import is_car_passing, cleanup as sensors_cleanup
from gate import open_gate, close_gate, cleanup as gate_cleanup
from database import get_allowed, update_allowed, add_unauthorized
import config

LOOP_DELAY = 0.5

def entry_worker():
    cap = cv2.VideoCapture(config.ENTRY_CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open entry camera {config.ENTRY_CAMERA_INDEX}")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(LOOP_DELAY)
            continue
        plate, make, model, color = detect_plate_and_vehicle(frame)
        if plate:
            allowed = {v['plate'] for v in get_allowed()}
            if plate in allowed:
                update_allowed(plate, make or '', model or '', color or '')
                open_gate()
                while not is_car_passing():
                    time.sleep(0.1)
                while is_car_passing():
                    time.sleep(0.1)
                close_gate()
            else:
                add_unauthorized(plate, make or '', model or '', color or '')
        time.sleep(LOOP_DELAY)

def exit_worker():
    cap = cv2.VideoCapture(config.EXIT_CAMERA_INDEX)
    while True:
        opened = False
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and detect_vehicle_exit(frame):
                opened = True
        if cv2.waitKey(1) & 0xFF == ord('e'):
            opened = True
        if opened:
            open_gate()
            while not is_car_passing():
                time.sleep(0.1)
            while is_car_passing():
                time.sleep(0.1)
            close_gate()
        time.sleep(LOOP_DELAY)

def main():
    try:
        threading.Thread(target=entry_worker, daemon=True).start()
        threading.Thread(target=exit_worker, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        sensors_cleanup()
        gate_cleanup()
        alpr_deinit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
