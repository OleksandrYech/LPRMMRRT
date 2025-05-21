# sheets.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import logging
from datetime import datetime

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константи
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'credentials.json'
YOUR_SPREADSHEET_ID = "1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4"

# Назви аркушів та діапазони
VEHICLES_SHEET_NAME = 'Vehicles' # Припускаємо, що так називається ваш аркуш
VEHICLES_RANGE = 'A3:B'       # Номери та останній в'їзд
UNAUTHORIZED_RANGE_WRITE = 'D3:E' # Куди писати неавторизовані спроби (починаючи з D3)

# Глобальна змінна для клієнта, щоб уникнути повторної автентифікації
SHEET_CLIENT = None

def _get_sheet_client():
    """
    Ініціалізує та повертає клієнт для роботи з Google Sheets.
    Використовує сервісний акаунт для автентифікації.
    """
    global SHEET_CLIENT
    if SHEET_CLIENT is None:
        try:
            # Використання oauth2client (традиційний спосіб)
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPES)
            SHEET_CLIENT = gspread.authorize(creds)

            
            logging.info("Successfully connected to Google Sheets API.")
        except FileNotFoundError:
            logging.error(f"Credentials file not found: {CREDENTIALS_FILE}. Please ensure it's in the correct path.")
            SHEET_CLIENT = None # Явно вказуємо, що клієнт не ініціалізований
        except Exception as e:
            logging.error(f"Failed to authorize with Google Sheets API: {e}")
            SHEET_CLIENT = None # Явно вказуємо, що клієнт не ініціалізований
    return SHEET_CLIENT

def read_sheet_data(sheet_title, range_name):
    """
    Читає дані з вказаного діапазону аркуша.

    Args:
        sheet_title (str): Назва аркуша.
        range_name (str): Діапазон для читання (наприклад, 'A1:B10').

    Returns:
        list: Список списків з даними, або None у разі помилки.
    """
    client = _get_sheet_client()
    if not client:
        return None
    try:
        sheet = client.open_by_key(YOUR_SPREADSHEET_ID).worksheet(VEHICLES_SHEET_NAME)
        
        data = sheet.get_all_values() # Поки що читаємо весь лист, потім можна оптимізувати до range_name
        if sheet_title == VEHICLES_SHEET_NAME and len(data) > 2:
             # Повертаємо лише дані з A3:B (стовпці 0 та 1)
            relevant_data = [[row[0], row[1] if len(row) > 1 else ""] for row in data[2:]]
            return relevant_data
        elif sheet_title == UNAUTHORIZED_SHEET_NAME:
             relevant_data = [[row[3], row[4] if len(row) > 4 else ""] for row in data[2:] if len(row) > 3]
             return relevant_data
        return data # Повертаємо всі дані для інших випадків або якщо логіка заголовків інша

    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"Spreadsheet not found. Check URL/ID and permissions.")
        return None
    except gspread.exceptions.WorksheetNotFound:
        logging.error(f"Worksheet '{sheet_title}' not found in the spreadsheet.")
        return None
    except Exception as e:
        logging.error(f"Error reading from sheet '{sheet_title}', range '{range_name}': {e}")
        return None

def find_vehicle_and_update_entry_time(plate_number):
    """
    Шукає номерний знак в аркуші 'Vehicles'.
    Якщо знайдено, оновлює час останнього в'їзду.

    Args:
        plate_number (str): Номерний знак для пошуку.

    Returns:
        bool: True, якщо номер знайдено та оновлено, False в іншому випадку.
    """
    client = _get_sheet_client()
    if not client:
        return False
    try:
        
        sheet = client.open_by_key(YOUR_SPREADSHEET_ID).worksheet(VEHICLES_SHEET_NAME) # ЗАМІНІТЬ ЦЕ

        # Отримуємо всі записи з стовпця А (номери) та В (час)
        
        list_of_lists = sheet.get_all_values()


        found = False
        for i, row in enumerate(list_of_lists):
            if i < 2: # Пропускаємо перші два рядки заголовків (A1, A2)
                continue
            if row and row[0] == plate_number: # Номер в стовпці A (індекс 0)

                sheet_row_index = i + 1
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.update_cell(sheet_row_index, 2, current_datetime) # Стовпець B - це 2-й стовпець
                logging.info(f"Vehicle {plate_number} found. Entry time updated to {current_datetime}.")
                found = True
                break
        
        if not found:
            logging.info(f"Vehicle {plate_number} not found in '{VEHICLES_SHEET_NAME}'.")
            
        return found

    except Exception as e:
        logging.error(f"Error finding/updating vehicle {plate_number}: {e}")
        return False

def add_unauthorized_attempt(plate_number):
    """
    Додає запис про неавторизовану спробу в'їзду.

    Args:
        plate_number (str): Розпізнаний номерний знак.

    Returns:
        bool: True, якщо запис успішно додано, False в іншому випадку.
    """
    client = _get_sheet_client()
    if not client:
        return False
    try:
   
        sheet = client.open_by_key(YOUR_SPREADSHEET_ID).worksheet(VEHICLES_SHEET_NAME) # ЗАМІНІТЬ ЦЕ

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        col_d_values = sheet.col_values(4) # 4-й стовпець (D)
        next_row_to_write = len(col_d_values) + 1
        
        # Якщо дані починаються з D3, то перші два рядки можуть бути заголовками або порожніми.
        # Якщо len(col_d_values) < 2, то пишемо в 3-й рядок.
        if next_row_to_write < 3:
            next_row_to_write = 3

        sheet.update_cell(next_row_to_write, 4, plate_number)      # Стовпець D
        sheet.update_cell(next_row_to_write, 5, current_datetime)  # Стовпець E
        
        logging.info(f"Unauthorized attempt by {plate_number} logged at {current_datetime}.")
        return True
    except Exception as e:
        logging.error(f"Error adding unauthorized attempt for {plate_number}: {e}")
        return False

# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    
    print("Тестування модуля sheets.py...")

    # Перевірка підключення (викличе _get_sheet_client неявно)
    if not _get_sheet_client():
        print("Не вдалося підключитися до Google Sheets. Перевірте лог та налаштування.")
    else:
        print("Підключення до Google Sheets успішне.")

        # Тест читання даних (припускаємо, що аркуш 'Vehicles' існує)
        # Потрібно вказати назву вашої таблиці та аркушів
        print(f"\nЧитання даних з аркуша '{VEHICLES_SHEET_NAME}':")
        vehicle_data = read_sheet_data(VEHICLES_SHEET_NAME, VEHICLES_RANGE)
        if vehicle_data is not None:
            print(f"Отримано {len(vehicle_data)} записів:")
            for row in vehicle_data[:5]: # Друкуємо перші 5 для прикладу
                print(row)
        else:
            print("Не вдалося прочитати дані.")

        # Тест пошуку та оновлення (замініть 'AB1234CE' на номер, який є у вашій таблиці)
        # test_plate_existing = "BI2680IB" # Номер, який ТОЧНО є у вашому списку A3:B
        # print(f"\nТест пошуку та оновлення для існуючого номера '{test_plate_existing}':")
        # if find_vehicle_and_update_entry_time(test_plate_existing):
        #     print(f"Успішно оновлено для {test_plate_existing}")
        # else:
        # print(f"Не вдалося оновити або знайти {test_plate_existing}")

        # Тест пошуку неіснуючого номера
        # test_plate_non_existing = "XX0000XX"
        # print(f"\nТест пошуку для неіснуючого номера '{test_plate_non_existing}':")
        # if not find_vehicle_and_update_entry_time(test_plate_non_existing):
        #     print(f"Номер {test_plate_non_existing} не знайдено, як і очікувалося.")
        # else:
        #     print(f"Помилка: номер {test_plate_non_existing} знайдено, хоча не мав би.")
            
        # Тест додавання неавторизованої спроби
        # test_unauthorized_plate = "ZZ9999ZZ"
        # print(f"\nТест додавання неавторизованої спроби для '{test_unauthorized_plate}':")
        # if add_unauthorized_attempt(test_unauthorized_plate):
        #     print(f"Успішно додано неавторизовану спробу для {test_unauthorized_plate}")
        # else:
        #     print(f"Не вдалося додати неавторизовану спробу для {test_unauthorized_plate}")

        # Ще одна спроба, щоб перевірити додавання в наступний рядок
        # test_unauthorized_plate_2 = "YY8888YY"
        # print(f"\nТест додавання другої неавторизованої спроби для '{test_unauthorized_plate_2}':")
        # if add_unauthorized_attempt(test_unauthorized_plate_2):
        #     print(f"Успішно додано неавторизовану спробу для {test_unauthorized_plate_2}")
        # else:
        #     print(f"Не вдалося додати неавторизовану спробу для {test_unauthorized_plate_2}")

    print("\nТестування модуля sheets.py завершено.")
