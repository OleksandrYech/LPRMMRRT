from unittest.mock import MagicMock

from sheets_client import SheetsClient


def test_is_allowed(mocker):
    client = SheetsClient.__new__(SheetsClient)  # не викликаємо __init__
    client.allowed = MagicMock()
    client.allowed.col_values.return_value = ["Номер", "Марка", "AA1234BB"]
    allowed, row = client.is_allowed("AA1234BB")
    assert allowed and row == 3