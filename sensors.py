# sensors.py
# ... (код класів ReedSwitch та UltrasonicSensor залишається таким самим, як я надавав раніше) ...

# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    logging.info("Тестування модуля sensors.py з оновленими пінами...")

    # --- Тест Геркона ---
    # Оновлений GPIO пін для геркона
    REED_PIN = 22  # Було 17
    reed_sensor = ReedSwitch(pin_number=REED_PIN, name=f"ReedSwitch_GPIO{REED_PIN}")

    if reed_sensor._device: # Перевірка, чи датчик ініціалізувався
        print(f"\nТестування геркона на GPIO{REED_PIN}:")
        print("Піднесіть/віддаліть магніт від геркона протягом наступних 10 секунд.")
        for i in range(20): # Тест протягом 10 секунд (20 * 0.5с)
            if reed_sensor.are_gates_open:
                # Нагадаємо: is_active (LOW) означає "ворота відкриті", якщо pull_up=True
                print(f"Стан геркона: ВОРОТА ВІДКРИТІ (контакт замкнений, GPIO{REED_PIN} is LOW)")
            else:
                print(f"Стан геркона: ВОРОТА ЗАКРИТІ (контакт розімкнений, GPIO{REED_PIN} is HIGH)")
            time.sleep(0.5)
        print("Тест геркона завершено.\n")
    else:
        print(f"Геркон на GPIO{REED_PIN} не ініціалізовано. Тест пропущено.")

    # --- Тест Ультразвукового датчика ---
    # Оновлені GPIO піни для ультразвукового датчика
    TRIGGER_PIN = 23  # Було 27
    ECHO_PIN = 24     # Було 22
    ultrasonic_sensor = UltrasonicSensor(trigger_pin=TRIGGER_PIN, echo_pin=ECHO_PIN, name=f"Ultrasonic_T{TRIGGER_PIN}_E{ECHO_PIN}")

    if ultrasonic_sensor._sensor: # Перевірка, чи датчик ініціалізувався
        print(f"Тестування ультразвукового датчика (Trigger: GPIO{TRIGGER_PIN}, Echo: GPIO{ECHO_PIN}):")
        print(f"  ВАЖЛИВО: Переконайтеся, що на Echo піні (GPIO{ECHO_PIN}) є дільник напруги (5V -> 3.3V)!")
        print(f"  Поріг під'їзду: {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_APPROACH:.2f} м")
        print(f"  Поріг проїзду (зона чиста): {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR:.2f} м")
        print(f"  Час підтвердження проїзду: {ultrasonic_sensor.PASS_CONFIRMATION_TIME:.1f} с")
        
        print("\nВимірювання відстані протягом 15 секунд:")
        start_time_ultrasonic_test = time.monotonic()
        vehicle_approached_logged = False
        vehicle_passed_logged = False # Для одноразового логування події проїзду

        while time.monotonic() - start_time_ultrasonic_test < 15:
            distance = ultrasonic_sensor.get_distance()
            if distance is not None:
                print(f"  Відстань: {distance:.2f} м")

                is_approaching = ultrasonic_sensor.is_vehicle_approaching()
                if is_approaching and not vehicle_approached_logged:
                    print(f"  !!! Автомобіль ПІД'ЇХАВ! (відстань < {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_APPROACH:.2f} м)")
                    # Можна закоментувати, якщо хочете бачити це повідомлення при кожному спрацюванні
                    # vehicle_approached_logged = True 
                # elif not is_approaching and vehicle_approached_logged:
                    # vehicle_approached_logged = False # Скидання, якщо авто від'їхало після під'їзду

                has_passed = ultrasonic_sensor.has_vehicle_passed()
                if has_passed:
                    if not vehicle_passed_logged:
                        print(f"  !!! Автомобіль ПРОЇХАВ! (зона чиста > {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR:.2f} м протягом {ultrasonic_sensor.PASS_CONFIRMATION_TIME:.1f} с)")
                        vehicle_passed_logged = True
                elif not has_passed and distance < ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR:
                    # Якщо авто знову близько або ще не підтвердило проїзд, скидаємо флаг
                    vehicle_passed_logged = False
            else:
                print(f"  Не вдалося отримати дані з ультразвукового датчика (GPIO{TRIGGER_PIN}/GPIO{ECHO_PIN}).")
            
            time.sleep(0.5) # Затримка між вимірами
        print("Тест ультразвукового датчика завершено.\n")
    else:
        print(f"Ультразвуковий датчик (Trig:GPIO{TRIGGER_PIN}, Echo:GPIO{ECHO_PIN}) не ініціалізовано. Тест пропущено.")

    logging.info("Тестування модуля sensors.py завершено.")