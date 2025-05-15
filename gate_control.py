"""
gate_control.py
---------------
Керує реле, герконовим та ультразвуковим датчиками.

GPIO піни виставіть під свої з’єднання!
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from gpiozero import Button, DistanceSensor, OutputDevice

_LOG = logging.getLogger(__name__)


class GateController:
    """Інкапсулює низькорівневу роботу з GPIO-пінами."""

    # ---- налаштування GPIO ---- #
    PIN_REED = 17            # MC-38 (геркон)
    PIN_RELAY_OPEN = 22      # реле "OPEN"
    PIN_RELAY_CLOSE = 23     # реле "CLOSE"
    TRIG = 5                 # AJ-SPO4M: TRIG
    ECHO = 6                 # AJ-SPO4M: ECHO

    ULTRASONIC_THRESHOLD_CM = 100   # коли менше — авто під’їхало

    def __init__(self) -> None:
        self.reed = Button(self.PIN_REED, pull_up=True, bounce_time=0.05)
        self.relay_open = OutputDevice(self.PIN_RELAY_OPEN, active_high=False, initial_value=False)
        self.relay_close = OutputDevice(self.PIN_RELAY_CLOSE, active_high=False, initial_value=False)
        self.ultra = DistanceSensor(echo=self.ECHO, trigger=self.TRIG, max_distance=4.0)

    # ---------- properties ---------- #

    @property
    def gate_is_open(self) -> bool:
        """True — ворота відкриті (геркон замкнений)."""
        return self.reed.is_pressed

    # ---------- high-level actions ---------- #

    def open_gate(self, pulse_ms: int = 300) -> None:
        _LOG.info("Opening gate")
        self.relay_open.on()
        time.sleep(pulse_ms / 1000)
        self.relay_open.off()

    def close_gate(self, pulse_ms: int = 300) -> None:
        _LOG.info("Closing gate")
        self.relay_close.on()
        time.sleep(pulse_ms / 1000)
        self.relay_close.off()

    # ---------- reactive helpers ---------- #

    def wait_vehicle(self, cb: Callable[[], None] | None = None) -> None:
        """
        Блокує виконання доки ультразвук не покаже об'єкт < threshold.
        Викликає cb() один раз коли подія трапилась.
        """
        _LOG.debug("Waiting vehicle under %scm…", self.ULTRASONIC_THRESHOLD_CM)
        self.ultra.wait_for_inactive()
        self.ultra.threshold_distance = self.ULTRASONIC_THRESHOLD_CM / 100
        self.ultra.wait_for_active()
        _LOG.debug("Vehicle detected by ultrasonic.")
        if cb:
            cb()
