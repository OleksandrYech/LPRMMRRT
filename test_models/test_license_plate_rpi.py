import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import tensorflow as tf  # Додано для TFLite

# --- Налаштування ---
MODEL_PT_PATH = "/home/pi/auto-gate/detection/models/ocr.pt"  # шлях до .pt
MODEL_ONNX_PATH = "/home/pi/auto-gate/detection/models/ocr.onnx"  # шлях до .onnx
# Вкажіть правильний шлях до вашої квантованої INT8 .tflite моделі
MODEL_TFLITE_INT8_PATH = "/home/pi/auto-gate/detection/models/ocr.tflite"  # Приклад шляху
IMAGE_PATH = "/home/pi/auto-gate/test_models/test.png"
CONFIDENCE_THRESHOLD = 0.12


def preprocess_image_for_tflite(img_bgr, input_details):
    """
    Перед обробка зображення для TFLite моделі.
    Включає зміну розміру, конвертацію кольору, нормалізацію та квантування (якщо потрібно).
    """
    # Очікувана форма та тип даних з моделі TFLite
    # Зазвичай input_details[0]['shape'] це [1, height, width, 3] (NHWC) для TFLite
    # або [1, 3, height, width] (NCHW)
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Визначаємо цільові розміри та формат каналів (NHWC чи NCHW)
    is_nchw = input_shape[1] == 3 and input_shape[3] != 3  # Грубе припущення
    if is_nchw:  # NCHW
        target_height = input_shape[2]
        target_width = input_shape[3]
    else:  # NHWC
        target_height = input_shape[1]
        target_width = input_shape[2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_width, target_height))

    # Нормалізація до [0, 1]
    input_data_float32 = img_resized.astype(np.float32) / 255.0

    if is_nchw:
        # HWC to CHW
        input_data_float32 = np.transpose(input_data_float32, (2, 0, 1))

    # Додавання осі батчу
    input_data_float32 = np.expand_dims(input_data_float32, axis=0)  # Форма (1, H, W, C) або (1, C, H, W)

    # Квантування, якщо модель очікує INT8/UINT8 на вході
    if input_dtype == np.int8 or input_dtype == np.uint8:
        quant_params = input_details[0]['quantization_parameters']
        scale = quant_params['scales'][0]
        zero_point = quant_params['zero_points'][0]
        input_tensor = (input_data_float32 / scale) + zero_point
        input_tensor = input_tensor.astype(input_dtype)
    else:  # Якщо модель очікує float32
        input_tensor = input_data_float32.astype(input_dtype)

    return input_tensor


def run_ocr_via_detection_test(model_path, image_path, model_name):
    """
    Завантажує модель (.pt або .onnx через YOLO), виконує розпізнавання символів,
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
        # YOLO() може завантажувати .pt та .onnx (якщо onnxruntime встановлено)
        model = YOLO(model_path)
        print("Модель успішно завантажена.")
        if not hasattr(model, 'names') or not model.names:
            print("ПОПЕРЕДЖЕННЯ: У моделі відсутні імена класів (model.names). "
                  "Результат розпізнавання символів може бути некоректним.")
    except Exception as e:
        print(f"Помилка завантаження моделі {model_path}: {e}")
        return

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Помилка завантаження зображення {image_path}")
            return
        print(f"Зображення {image_path} успішно завантажено (для OCR).")

        # Прогрів моделі
        _ = model(img_bgr, verbose=False, conf=CONFIDENCE_THRESHOLD)

        num_runs = 5
        inference_times = []

        recognized_plate_string_last_run = "НЕ РОЗПІЗНАНО"
        avg_char_confidence_last_run = 0.0
        char_details_last_run = []

        print(f"Запуск розпізнавання символів ({num_runs} разів)...")
        for i in range(num_runs):
            start_time = time.perf_counter()
            results_list = model(img_bgr, verbose=False, conf=CONFIDENCE_THRESHOLD)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            if i == num_runs - 1 and results_list:
                results = results_list[0]  # Перший результат з батчу
                detected_characters_info = []

                if results.boxes:
                    for box in results.boxes:
                        confidence = box.conf.item()
                        cls_id = int(box.cls.item())

                        character_str = "UNK"
                        if model.names and cls_id < len(model.names):
                            character_str = model.names[cls_id]
                        else:
                            print(f"ПОПЕРЕДЖЕННЯ: Не знайдено ім'я для класу ID {cls_id}. Використано 'UNK'.")

                        x_center = box.xywh.squeeze().tolist()[0]

                        detected_characters_info.append({
                            "x": x_center,
                            "char": character_str,
                            "conf": confidence
                        })

                detected_characters_info.sort(key=lambda item: item["x"])
                recognized_plate_string_last_run = "".join([item["char"] for item in detected_characters_info])
                char_confidences = [item["conf"] for item in detected_characters_info]

                if char_confidences:
                    avg_char_confidence_last_run = np.mean(char_confidences)
                    char_details_last_run = [f"{item['char']}({item['conf']:.2f})" for item in detected_characters_info]

        avg_inference_time = np.mean(inference_times)
        print(f"\nРезультати для {model_name}:")
        print(f"  Середній час розпізнавання: {avg_inference_time:.4f} секунд")

        if recognized_plate_string_last_run != "НЕ РОЗПІЗНАНО" and char_details_last_run:
            print(f"  Розпізнаний номерний знак (останній запуск): '{recognized_plate_string_last_run}'")
            print(f"  Середня впевненість символів (останній запуск): {avg_char_confidence_last_run:.4f}")
        else:
            print(f"  Символи не були розпізнані (поріг: {CONFIDENCE_THRESHOLD}).")

    except Exception as e:
        print(f"Помилка під час розпізнавання моделлю {model_path}: {e}")
        import traceback
        traceback.print_exc()


def run_tflite_ocr_test(model_path, image_path, model_name, class_names):
    """
    Завантажує модель TFLite, виконує розпізнавання символів на зображенні,
    збирає їх у рядок та виводить результати.
    """
    print(f"\n--- Тестування моделі: {model_name} ({model_path}) ---")

    if not class_names:
        print("ПОМИЛКА: Список імен класів не надано для TFLite моделі.")
        return
    if not os.path.exists(model_path):
        print(f"ПОМИЛКА: Файл моделі TFLite не знайдено: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"ПОМИЛКА: Файл зображення не знайдено: {image_path}")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Модель TFLite успішно завантажена.")
        # print("Input details:", input_details) # Для відладки
        # print("Output details:", output_details) # Для відладки

    except Exception as e:
        print(f"Помилка завантаження моделі TFLite {model_path}: {e}")
        return

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Помилка завантаження зображення {image_path}")
            return
        print(f"Зображення {image_path} успішно завантажено (для OCR).")

        # Перед обробка зображення для TFLite
        input_tensor = preprocess_image_for_tflite(img_bgr, input_details)

        # Прогрів моделі
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])  # Отримати результат, щоб завершити прогрів

        num_runs = 5
        inference_times = []
        recognized_plate_string_last_run = "НЕ РОЗПІЗНАНО"
        avg_char_confidence_last_run = 0.0
        char_details_last_run = []

        # Отримуємо розміри оригінального зображення для масштабування рамок (якщо потрібно)
        original_height, original_width = img_bgr.shape[:2]

        # Отримуємо розміри входу моделі для денормалізації координат, якщо вони нормалізовані моделлю
        model_input_shape = input_details[0]['shape']
        # Припускаємо NHWC або NCHW
        if model_input_shape[1] == 3 and len(model_input_shape) == 4:  # NCHW
            model_input_height = model_input_shape[2]
            model_input_width = model_input_shape[3]
        elif model_input_shape[3] == 3 and len(model_input_shape) == 4:  # NHWC
            model_input_height = model_input_shape[1]
            model_input_width = model_input_shape[2]
        else:  # Невідомий або непідтримуваний формат
            print(
                f"ПОПЕРЕДЖЕННЯ: Не вдалося визначити розміри входу моделі з форми {model_input_shape}. Масштабування рамок може бути неточним.")
            model_input_height, model_input_width = target_height, target_width  # Резервний варіант

        print(f"Запуск розпізнавання символів TFLite ({num_runs} разів)...")
        for i in range(num_runs):
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            # Припускаємо, що output_details[0] - це основний вихід з детекціями
            # Форма вихідного тензора може бути [1, N, 6] (x1,y1,x2,y2, conf, cls_id)
            # або [1, num_predictions, num_attributes] (наприклад, [1, 8400, 40] для YOLO)
            # Це залежить від того, як була експортована TFLite модель (з NMS чи без)
            raw_detections = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            if i == num_runs - 1:  # Обробляємо детально тільки останній запуск
                detected_characters_info = []

                # Обробка вихідних даних - ЦЕЙ БЛОК ПОТРІБНО АДАПТУВАТИ ПІД ВАШУ МОДЕЛЬ TFLITE
                # Варіант 1: Якщо вихід [1, N, 6] де N - кількість детекцій,
                # а 6 -> [x1, y1, x2, y2, confidence, class_id]
                # І координати нормалізовані до [0,1] відносно розміру входу моделі
                if raw_detections.shape[2] == 6 and len(raw_detections.shape) == 3:  # Приклад для [1, N, 6]
                    detections = raw_detections[0]  # Беремо перший (і єдиний) батч
                    for detection in detections:
                        x1, y1, x2, y2, confidence, cls_id_float = detection
                        if confidence >= CONFIDENCE_THRESHOLD:
                            cls_id = int(cls_id_float)
                            character_str = "UNK"
                            if cls_id < len(class_names):
                                character_str = class_names[cls_id]
                            else:
                                print(f"ПОПЕРЕДЖЕННЯ: Не знайдено ім'я для класу ID {cls_id}. Використано 'UNK'.")

                            # Денормалізація координат та отримання центру X
                            # Якщо координати з TFLite нормалізовані до розміру входу моделі (model_input_width)
                            abs_x1 = x1 * original_width / model_input_width
                            abs_x2 = x2 * original_width / model_input_width
                            x_center = (abs_x1 + abs_x2) / 2.0
                            # Або якщо вони вже в абсолютних координатах відносно входу моделі:
                            # x_center = ((x1 + x2) / 2.0) * (original_width / model_input_width)

                            detected_characters_info.append({
                                "x": x_center,  # Використовуємо масштабований x_center
                                "char": character_str,
                                "conf": confidence
                            })
                # Варіант 2: Якщо вихід [1, num_attributes, num_predictions] як у YOLO (напр. [1,40,8400])
                # Цей варіант потребує складнішої обробки: декодування рамок, NMS.
                # Для прикладу, припустимо, що Ваша TFLite модель має вбудований NMS
                # і вихідний формат простий, як у Варіанті 1.
                # Якщо ні, цей блок потрібно буде значно переписати.
                # Див. документацію експорту Ultralytics для формату виходу TFLite.

                else:
                    print(f"ПОПЕРЕДЖЕННЯ: Невідомий або непідтримуваний формат виходу TFLite: {raw_detections.shape}")

                detected_characters_info.sort(key=lambda item: item["x"])
                recognized_plate_string_last_run = "".join([item["char"] for item in detected_characters_info])
                char_confidences = [item["conf"] for item in detected_characters_info]

                if char_confidences:
                    avg_char_confidence_last_run = np.mean(char_confidences)
                    char_details_last_run = [f"{item['char']}({item['conf']:.2f})" for item in detected_characters_info]

        avg_inference_time = np.mean(inference_times)
        print(f"\nРезультати для {model_name}:")
        print(f"  Середній час розпізнавання TFLite: {avg_inference_time:.4f} секунд")

        if recognized_plate_string_last_run != "НЕ РОЗПІЗНАНО" and char_details_last_run:
            print(f"  Розпізнаний номерний знак (останній запуск): '{recognized_plate_string_last_run}'")
            print(f"  Середня впевненість символів (останній запуск): {avg_char_confidence_last_run:.4f}")
        else:
            print(f"  Символи не були розпізнані (поріг: {CONFIDENCE_THRESHOLD}).")

    except Exception as e:
        print(f"Помилка під час розпізнавання моделлю TFLite {model_path}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Завантажуємо імена класів один раз з .pt моделі
    # Це потрібно, оскільки TFLite моделі зазвичай не зберігають імена класів.
    class_names_list = []
    try:
        pt_model_for_names = YOLO(MODEL_PT_PATH)
        if hasattr(pt_model_for_names, 'names') and pt_model_for_names.names:
            class_names_list = pt_model_for_names.names
            print(f"Імена класів завантажено з {MODEL_PT_PATH}: {class_names_list}")
        else:
            print(f"ПОМИЛКА: Не вдалося завантажити імена класів з {MODEL_PT_PATH}. Перевірте модель.")
            # Якщо імена класів відомі, можна їх задати вручну:
            # class_names_list = ['0', '1', '2', ..., 'X', 'Y', 'Z'] # Приклад
    except Exception as e:
        print(f"ПОМИЛКА при завантаженні імен класів з {MODEL_PT_PATH}: {e}")
        print("Будь ласка, переконайтеся, що MODEL_PT_PATH правильний або задайте class_names_list вручну.")

    run_ocr_via_detection_test(MODEL_PT_PATH, IMAGE_PATH, "PyTorch (.pt)")
    run_ocr_via_detection_test(MODEL_ONNX_PATH, IMAGE_PATH, "ONNX (.onnx)")

    # Тестування .tflite (INT8) моделі
    if class_names_list:  # Запускаємо тест TFLite тільки якщо є імена класів
        run_tflite_ocr_test(MODEL_TFLITE_INT8_PATH, IMAGE_PATH, "TFLite INT8", class_names_list)
    else:
        print(f"\nТестування TFLite моделі ({MODEL_TFLITE_INT8_PATH}) пропущено через відсутність імен класів.")

    print("\n--- Тестування завершено ---")