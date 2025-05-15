"""
sheets_client.py
----------------
Google Sheets API через gspread.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import gspread
from google.oauth2.service_account import Credentials

_LOG = logging.getLogger(__name__)


class SheetsClient:
    SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)

    def __init__(self, creds_json: Path | str, spreadsheet_key: str) -> None:
        creds = Credentials.from_service_account_file(creds_json, scopes=self.SCOPES)
        gc = gspread.authorize(creds)
        self.sh = gc.open_by_key(spreadsheet_key)
        self.allowed = self.sh.worksheet("Vehicles")   # один лист з двома зонами
        _LOG.info("Google Sheets ready")

    # ---------- API ---------- #

    def is_allowed(self, plate: str) -> Tuple[bool, Optional[int]]:
        """Повертає (True, row_index) якщо номер у ALLOWED."""
        col_a = self.allowed.col_values(1)[2:]  # пропускаємо header
        try:
            idx = col_a.index(plate) + 3        # A3 == рядок 3
            return True, idx
        except ValueError:
            return False, None

    def update_last_entry(self, row_idx: int) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.allowed.update_cell(row_idx, 5, ts)  # колонка E
        _LOG.info("ALLOWED row %s updated timestamp %s", row_idx, ts)

    def log_unauthorized(self, data: Dict[str, str]) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [data["plate"], data["make"], data["model"], data["color"], ts]
        self.allowed.append_row(row, table_range="G3:K")  # зона UNAUTHORIZED
        _LOG.warning("Logged UNAUTHORIZED access: %s", row)
