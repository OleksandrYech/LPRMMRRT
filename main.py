import cv2
import threading
import time

import config
from camera import LibcameraCapture                       # ← новий імпорт
from database import get_allowed, update_allowed, add_unauthorized
from detection import (
    detect_plate_and_vehicle,
    detect_vehicle_exit,
    deinit as alpr_deinit,
)
from sensors import is_car_passing, cleanup as sensors_cleanup
from gate import open_gate, close_gate, cleanup as gate_cleanup

LOOP_DELAY = 0.5  # seconds


def entry_worker() -> None:
    """ALPR + логіка в’їзду."""
    cap = LibcameraCapture(                              # ← заміна
        camera_index=config.ENTRY_CAMERA_INDEX,
        width=getattr(config, "CAM_WIDTH", 640),
        height=getattr(config, "CAM_HEIGHT", 480),
    )
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open entry camera {config.ENTRY_CAMERA_INDEX}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(LOOP_DELAY)
            continue

        plate, make, model, color = detect_plate_and_vehicle(frame)
        if plate:
            allowed = {v["plate"] for v in get_allowed()}
            if plate in allowed:
                update_allowed(plate, make or "", model or "", color or "")
                open_gate()
                while not is_car_passing():
                    time.sleep(0.1)
                while is_car_passing():
                    time.sleep(0.1)
                close_gate()
            else:
                add_unauthorized(plate, make or "", model or "", color or "")
        time.sleep(LOOP_DELAY)


def exit_worker() -> None:
    """Автоматичний/ручний виїзд."""
    cap = LibcameraCapture(                              # ← заміна
        camera_index=config.EXIT_CAMERA_INDEX,
        width=getattr(config, "CAM_WIDTH", 640),
        height=getattr(config, "CAM_HEIGHT", 480),
    )
    if not cap.isOpened():
        print(f"[warning] cannot open exit camera {config.EXIT_CAMERA_INDEX}")

    while True:
        opened = False
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and detect_vehicle_exit(frame):
                opened = True

        if cv2.waitKey(1) & 0xFF == ord("e"):
            opened = True

        if opened:
            open_gate()
            while not is_car_passing():
                time.sleep(0.1)
            while is_car_passing():
                time.sleep(0.1)
            close_gate()

        time.sleep(LOOP_DELAY)


def main() -> None:
    try:
        threading.Thread(target=entry_worker, daemon=True).start()
        threading.Thread(target=exit_worker, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        sensors_cleanup()
        gate_cleanup()
        alpr_deinit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
