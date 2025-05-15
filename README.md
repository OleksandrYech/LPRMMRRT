# LPRMMRRT — Raspberry Pi 5 / 32-bit

Готова система розпізнавання номерів і керування ворітьми.

## Швидкий старт
1. `sudo apt install python3-venv git`  
2. `git clone https://github.com/OleksandrYech/LPRMMRRT`  
3. `python3 -m venv venv && source venv/bin/activate`  
4. `pip install -r requirements.txt`  
5. `python test.py` — перевірка заліза  
6. `python main.py`

## Схема підключення
* **BCM 17** — реле «open», активний LOW  
* **BCM 27** — реле «close», активний LOW  
* **BCM 23** — trig AJ-SR04M  
* **BCM 24** — echo AJ-SR04M  
* **BCM 22** — геркон (NC → GND)  

## Ліцензія
MIT
