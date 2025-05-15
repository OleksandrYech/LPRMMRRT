import datetime
from typing import List, Dict

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

import config

# https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0
# Google Sheets API Scope
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def _get_sheets_service():
    """
    Ініціалізація клієнту Google Sheets API, використовуючи облікові дані сервісного акаунта.
    """
    creds = Credentials.from_service_account_file(
        config.SERVICE_ACCOUNT_FILE,
        scopes=SCOPES
    )
    service = build('sheets', 'v4', credentials=creds)
    return service.spreadsheets()


def get_allowed() -> List[Dict[str, str]]:
    """
    Отримує список дозволених транспортних засобів з Google Таблиць.
    Повертає список словника з ключами: номер, марка, модель, колір, останній_вхід.
    """
    sheet = _get_sheets_service()
    result = sheet.values().get(
        spreadsheetId=config.GOOGLE_SHEET_ID,
        range=config.GOOGLE_SHEET_RANGE_ALLOWED
    ).execute()
    rows = result.get('values', [])
    vehicles: List[Dict[str, str]] = []
    for row in rows:
        # Ensure row has 5 columns
        row += [''] * (5 - len(row))
        vehicles.append({
            'plate': row[0],
            'make': row[1],
            'model': row[2],
            'color': row[3],
            'last_entry': row[4]
        })
    return vehicles


def get_unauthorized() -> List[Dict[str, str]]:
    """
    Отримує список несанкціонованих автомобілів з Google Таблиць.
    Повертає список диктів з ключами: номер, марка, модель, колір, дата спроби.
    """
    sheet = _get_sheets_service()
    result = sheet.values().get(
        spreadsheetId=config.GOOGLE_SHEET_ID,
        range=config.GOOGLE_SHEET_RANGE_UNAUTHORIZED
    ).execute()
    rows = result.get('values', [])
    vehicles: List[Dict[str, str]] = []
    for row in rows:
        row += [''] * (5 - len(row))
        vehicles.append({
            'plate': row[0],
            'make': row[1],
            'model': row[2],
            'color': row[3],
            'attempt_date': row[4]
        })
    return vehicles


def update_allowed(plate: str, make: str, model: str, color: str, last_entry: str = None) -> bool:
    """
    Оновлює існуючий дозволений в'їзд транспортного засобу, ідентифікований за номером.
    Оновлює марку, модель, колір та останній_запис (за замовчуванням - поточна мітка часу).
    Повертає True, якщо запис знайдено та оновлено, False в іншому випадку.
    """
    entries = get_allowed()
    for idx, entry in enumerate(entries):
        if entry['plate'] == plate:
            timestamp = last_entry or datetime.datetime.now().isoformat()
            values = [[plate, make, model, color, timestamp]]
            # Calculate A:E range for the specific row (row index + header offset)
            sheet_name = config.GOOGLE_SHEET_RANGE_ALLOWED.split('!')[0]
            row_number = idx + 3  # data starts from row 3
            range_name = f"{sheet_name}!A{row_number}:E{row_number}"
            _get_sheets_service().values().update(
                spreadsheetId=config.GOOGLE_SHEET_ID,
                range=range_name,
                valueInputOption='RAW',
                body={'values': values}
            ).execute()
            return True
    return False


def add_unauthorized(plate: str, make: str, model: str, color: str, attempt_date: str = None) -> None:
    """
    Додає новий запис до несанкціонованих транспортних засобів з поточною або наданою міткою часу.
    """
    timestamp = attempt_date or datetime.datetime.now().isoformat()
    values = [[plate, make, model, color, timestamp]]
    _get_sheets_service().values().append(
        spreadsheetId=config.GOOGLE_SHEET_ID,
        range=config.GOOGLE_SHEET_RANGE_UNAUTHORIZED,
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body={'values': values}
    ).execute()
    