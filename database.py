"""
database.py
Взаємодія з Google Sheets.
"""

import datetime as _dt
from typing import List, Dict
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import config

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def _service():
    creds = Credentials.from_service_account_file(config.SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds).spreadsheets()

def get_allowed() -> List[Dict[str, str]]:
    rsp = _service().values().get(
        spreadsheetId=config.GOOGLE_SHEET_ID,
        range=config.GOOGLE_SHEET_RANGE_ALLOWED
    ).execute()
    rows = rsp.get('values', [])
    return [
        dict(zip(['plate', 'make', 'model', 'color', 'last_entry'], row + [''] * 5))
        for row in rows
    ]

def update_allowed(plate: str, make: str, model: str, color: str) -> bool:
    entries = get_allowed()
    for idx, entry in enumerate(entries):
        if entry['plate'] == plate:
            values = [[plate, make, model, color, _dt.datetime.now().isoformat()]]
            row = idx + 3
            rng = f"{config.GOOGLE_SHEET_RANGE_ALLOWED.split('!')[0]}!A{row}:E{row}"
            _service().values().update(
                spreadsheetId=config.GOOGLE_SHEET_ID,
                range=rng,
                valueInputOption='RAW',
                body={'values': values}
            ).execute()
            return True
    return False

def add_unauthorized(plate: str, make: str, model: str, color: str) -> None:
    values = [[plate, make, model, color, _dt.datetime.now().isoformat()]]
    _service().values().append(
        spreadsheetId=config.GOOGLE_SHEET_ID,
        range=config.GOOGLE_SHEET_RANGE_UNAUTHORIZED,
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body={'values': values}
    ).execute()
