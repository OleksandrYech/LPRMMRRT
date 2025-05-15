# Auto-Gate on Raspberry Pi 5

Система автоматичного відкриття воріт із розпізнаванням номерних знаків.

## Встановлення

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv libatlas-base-dev
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
