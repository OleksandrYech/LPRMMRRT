"""
Швидке ручне тестування GPIO та ALPR.
Запускати після збирання віртуального середовища: `python test.py`
"""

import time

import cv2

import sensors
from detection import recognize_bgr

print("⏱  Тест реле...")
sensors.relay_open()
time.sleep(1)
sensors.relay_close()

print("⏱  Тест ультразвуку...")
dist = sensors.get_distance_cm()
print(f"Відстань: {dist:.1f} см")

print("⏱  Тест ALPR (1 кадр з камери)...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
ok, frame = cap.read()
if ok:
    r = recognize_bgr(frame)
    print("ALPR:", r or "номер не знайдено")
else:
    print("Камеру не знайдено")

print("✅  Тести завершено")
sensors.cleanup()
