import cv2
import time
import numpy as np
from ultralytics import YOLO
import os

# --- Налаштування ---
MODEL_PT_PATH = "/home/pi/auto-gate/detection/models/ocr.pt"  # шлях до .pt
MODEL_ONNX_PATH = "/home/pi/auto-gate/detection/models/ocr.onnx"  # шлях до .onnx
IMAGE_PATH = "/home/pi/auto-gate/test_models/test.png"
CONFIDENCE_THRESHOLD = 0.12


def run_ocr_via_detection_test(model_path, image_path, model_name):
    """
    Завантажує модель, виконує розпізнавання символів на зображенні номерного знаку,
    збирає їх у рядок та виводить результати.
    """
    print(f"\n--- Тестування моделі: {model_name} ({model_path}) ---")

    if not os.path.exists(model_path):
        print(f"ПОМИЛКА: Файл моделі не знайдено: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"ПОМИЛКА: Файл зображення не знайдено: {image_path}")
        return

    try:
        model = YOLO(model_path)
        print("Модель успішно завантажена.")
        if not model.names:
            print("ПОПЕРЕДЖЕННЯ: У моделі відсутні імена класів (model.names). "
                  "Результат розпізнавання символів може бути некоректним.")
            # Можна завершити виконання або спробувати продовжити з ID класів
    except Exception as e:
        print(f"Помилка завантаження моделі {model_path}: {e}")
        return

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Помилка завантаження зображення {image_path}")
            return
        print(f"Зображення {image_path} успішно завантажено (для OCR).")

        # Прогрів моделі
        _ = model(img, verbose=False, conf=CONFIDENCE_THRESHOLD)

        num_runs = 5
        inference_times = []

        # Змінні для зберігання результатів останнього прогону
        recognized_plate_string_last_run = "НЕ РОЗПІЗНАНО"
        avg_char_confidence_last_run = 0.0
        char_details_last_run = []

        print(f"Запуск розпізнавання символів ({num_runs} разів)...")
        for i in range(num_runs):
            start_time = time.perf_counter()
            results_list = model(img, verbose=False, conf=CONFIDENCE_THRESHOLD)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            if i == num_runs - 1 and results_list:  # Обробляємо детально тільки останній запуск
                results = results_list[0]
                detected_characters_info = []  # Список для (x_координата, символ_str, впевненість)

                if results.boxes:
                    for box in results.boxes:
                        confidence = box.conf.item()
                        cls_id = int(box.cls.item())

                        character_str = "UNK"  # Якщо ім'я класу не знайдено
                        if model.names and cls_id < len(model.names):
                            character_str = model.names[cls_id]
                        else:
                            print(f"ПОПЕРЕДЖЕННЯ: Не знайдено ім'я для класу ID {cls_id}. Використано 'UNK'.")

                        # Використовуємо x-координату центру рамки для сортування
                        x_center = box.xywh.squeeze().tolist()[0]
                        # Або ліву x-координату: x_coordinate = box.xyxy.squeeze().tolist()[0]

                        detected_characters_info.append({
                            "x": x_center,
                            "char": character_str,
                            "conf": confidence
                        })

                # Сортуємо виявлені символи за їхньою x-координатою
                detected_characters_info.sort(key=lambda item: item["x"])

                recognized_plate_string_last_run = "".join([item["char"] for item in detected_characters_info])
                char_confidences = [item["conf"] for item in detected_characters_info]

                if char_confidences:
                    avg_char_confidence_last_run = np.mean(char_confidences)
                    char_details_last_run = [f"{item['char']}({item['conf']:.2f})" for item in detected_characters_info]

        avg_inference_time = np.mean(inference_times)

        print(f"\nРезультати для {model_name}:")
        print(f"  Середній час розпізнавання (на обрізаному зображенні): {avg_inference_time:.4f} секунд")

        if recognized_plate_string_last_run != "НЕ РОЗПІЗНАНО" and char_details_last_run:
            print(f"  Розпізнаний номерний знак (останній запуск): '{recognized_plate_string_last_run}'")
            print(f"  Середня впевненість символів (останній запуск): {avg_char_confidence_last_run:.4f}")
            # print(f"  Символи (впевненість): {' '.join(char_details_last_run)}") # Розкоментуйте для більшої деталізації
        else:
            print(
                f"  Символи не були розпізнані або виявлені з достатньою впевненістю (поріг: {CONFIDENCE_THRESHOLD}).")

    except Exception as e:
        print(f"Помилка під час розпізнавання моделлю {model_path}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    run_ocr_via_detection_test(MODEL_PT_PATH, IMAGE_PATH, "PyTorch (.pt)")

    # Тестування .onnx моделі
    run_ocr_via_detection_test(MODEL_ONNX_PATH, IMAGE_PATH, "ONNX (.onnx)")

    print("\n--- Тестування завершено ---")