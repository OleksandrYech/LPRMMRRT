"""main.py
Entry point for the LPRMMRRT system (License‑Plate Recognition, Monitoring,
Management, and Remote Relay Trigger).

The module launches two daemon threads:

* **entry_worker** ― watches the *entry* camera and automatically opens the
  gate when a recognised plate is found in the *allowed* database.
* **exit_worker** ― watches the *exit* camera and opens the gate when a car
  is detected at the exit loop, or when an operator presses the `E` key in the
  OpenCV window (manual override).

Both threads share the same helper utilities (detectors, sensors, relay
control) and clean up all hardware resources on shutdown (Ctrl‑C).

The file replaces direct ``cv2.VideoCapture`` usage with :class:`camera.LibcameraCapture`.
This ensures the program runs equally well on Raspberry Pi OS using the
``libcamera`` stack or on any Linux PC where only a USB‑UVC camera is
available.

Floating parameters like camera indices and resolution are taken from
``config.py`` and can also be overridden via environment variables.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2

# Project‑local imports ────────────────────────────────────────────────────
import config
from camera import LibcameraCapture
from database import add_unauthorized, get_allowed, update_allowed
from detection import (
    deinit as alpr_deinit,
    detect_plate_and_vehicle,
    detect_vehicle_exit,
)
from gate import close_gate, open_gate, cleanup as gate_cleanup
from sensors import (
    cleanup as sensors_cleanup,
    is_car_passing,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = (
    logging.getLevelName(config.LOG_LEVEL)
    if hasattr(config, "LOG_LEVEL")
    else "INFO"
)
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(Path(__file__).stem)

# ---------------------------------------------------------------------------
# Runtime constants (may be tuned through config.py).
# ---------------------------------------------------------------------------
# Delay (in seconds) between consecutive iterations inside the worker loops.
LOOP_DELAY: float = getattr(config, "LOOP_DELAY", 0.5)


# =============================================================================
# Worker threads
# =============================================================================

def entry_worker() -> None:
    """Recognises vehicles approaching the ENTRY gate and opens it automatically."""
    cam = LibcameraCapture(
        camera_index=config.ENTRY_CAMERA_INDEX,
        width=getattr(config, "CAM_WIDTH", 640),
        height=getattr(config, "CAM_HEIGHT", 480),
    )

    if not cam.isOpened():
        logger.error("Cannot open entry camera (index %s)", config.ENTRY_CAMERA_INDEX)
        return

    allowed_cache = {v["plate"] for v in get_allowed()}  # Minimises DB hits.
    logger.info("Entry worker started with %d allowed plates cached", len(allowed_cache))

    while True:
        ok, frame = cam.read()
        if not ok:
            logger.warning("Entry camera read failed – retrying in %.1fs", LOOP_DELAY)
            time.sleep(LOOP_DELAY)
            continue

        plate, make, model, color = detect_plate_and_vehicle(frame)
        if plate:
            logger.debug("Detected plate '%s' on entry", plate)
            # Refresh cache every detection to accommodate external DB changes.
            if plate not in allowed_cache:
                allowed_cache = {v["plate"] for v in get_allowed()}

            if plate in allowed_cache:
                logger.info("Plate '%s' is allowed – opening gate", plate)
                update_allowed(plate, make or "", model or "", color or "")
                _cycle_gate()
            else:
                logger.warning("Plate '%s' is NOT allowed – logging as unauthorized", plate)
                add_unauthorized(plate, make or "", model or "", color or "")

        time.sleep(LOOP_DELAY)


def exit_worker() -> None:
    """Detects cars at the EXIT gate and opens it either automatically or manually."""
    cam = LibcameraCapture(
        camera_index=config.EXIT_CAMERA_INDEX,
        width=getattr(config, "CAM_WIDTH", 640),
        height=getattr(config, "CAM_HEIGHT", 480),
    )

    if not cam.isOpened():
        logger.warning("Cannot open exit camera (index %s) – exit recognition disabled", config.EXIT_CAMERA_INDEX)

    logger.info("Exit worker started (%s)", "automatic+manual" if cam.isOpened() else "manual only")

    while True:
        gate_should_open = False

        if cam.isOpened():
            ok, frame = cam.read()
            if ok and detect_vehicle_exit(frame):
                logger.debug("Vehicle detected at exit camera – opening gate")
                gate_should_open = True

        # Manual override – operator presses 'e' in any OpenCV window.
        if cv2.waitKey(1) & 0xFF == ord("e"):
            logger.info("Operator override – opening exit gate")
            gate_should_open = True

        if gate_should_open:
            _cycle_gate()

        time.sleep(LOOP_DELAY)


# =============================================================================
# Helper utilities
# =============================================================================

def _cycle_gate() -> None:
    """Open, wait until the car passes, then close the gate."""
    open_gate()

    # Wait until the front bumper triggers the *in* loop sensor.
    while not is_car_passing():
        time.sleep(0.05)

    # Wait until the back bumper clears the *out* loop sensor.
    while is_car_passing():
        time.sleep(0.05)

    close_gate()


# =============================================================================
# Graceful shutdown handling
# =============================================================================

def _cleanup() -> None:
    """Release all hardware resources and stop background threads gracefully."""
    logger.info("Cleaning up GPIO, ALPR, and OpenCV resources…")
    sensors_cleanup()
    gate_cleanup()
    alpr_deinit()
    cv2.destroyAllWindows()
    logger.info("Shutdown complete – bye!")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:  # noqa: D401 – imperatively named entry‑point.
    """Initialise and run the system until interrupted by the operator."""
    try:
        logger.info("Starting LPRMMRRT … press Ctrl‑C to stop")
        threading.Thread(target=entry_worker, daemon=True, name="entry_worker").start()
        threading.Thread(target=exit_worker, daemon=True, name="exit_worker").start()

        # Keep the main thread alive while workers run as daemons.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received – shutting down…")
    finally:
        _cleanup()


if __name__ == "__main__":  # pragma: no cover
    main()
