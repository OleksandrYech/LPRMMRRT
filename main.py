"""
main.py
-------
Головний цикл: стежить за подіями та координує ALPR → Sheets → Gate.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from alpr_service import AlprService
from gate_control import GateController
from sheets_client import SheetsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s %(name)18s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOG = logging.getLogger("main")

# ---------- конфіг ---------- #
CREDS = Path("/home/pi/.config/google/service-account.json")
SPREADSHEET_KEY = "YOUR_SPREADSHEET_KEY"
GATE_CLOSE_DELAY = 20          # сек — після відкриття зачекати й закрити


class GateSystem:
    def __init__(self) -> None:
        self.sheets = SheetsClient(CREDS, SPREADSHEET_KEY)
        self.alpr = AlprService()
        self.gpio = GateController()

    # ------------ tasks ------------ #

    async def entry_loop(self) -> None:
        """В’їзд: чекаємо авто, робимо ALPR, вирішуємо долю."""
        while True:
            self.gpio.wait_vehicle()
            data = self.alpr.detect_vehicle()
            if not data:
                _LOG.info("Nothing recognized — ignoring.")
                continue

            allowed, row = self.sheets.is_allowed(data["plate"])
            if allowed:
                _LOG.info("Access granted to %s", data["plate"])
                self.gpio.open_gate()
                self.sheets.update_last_entry(row)
                await asyncio.sleep(GATE_CLOSE_DELAY)
                self.gpio.close_gate()
            else:
                _LOG.warning("Access denied to %s", data["plate"])
                self.sheets.log_unauthorized(data)

    async def exit_loop(self) -> None:
        """Виїзд: лише ультразвук => відкрити + таймаут."""
        while True:
            self.gpio.wait_vehicle()
            _LOG.info("Exit detected")
            self.gpio.open_gate()
            await asyncio.sleep(GATE_CLOSE_DELAY)
            self.gpio.close_gate()

    # ------------ top-level ------------ #

    async def run(self) -> None:
        await asyncio.gather(self.entry_loop(), self.exit_loop())


# ---------- запуск ---------- #

async def _amain() -> None:
    gs = GateSystem()
    await gs.run()


def main() -> None:
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(_amain())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
