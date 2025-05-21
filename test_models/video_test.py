import cv2
import numpy as np
import time
import argparse
import os

# --- Імпорти бібліотек для моделей ---
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Попередження: Бібліотека 'ultralytics' не встановлена. .pt моделі не будуть доступні.")

try:
    import onnxruntime

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Попередження: Бібліотека 'onnxruntime' не встановлена. .onnx моделі не будуть доступні.")

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Попередження: Бібліотека 'tensorflow' не встановлена. .tflite моделі не будуть доступні.")

# --- Глобальні налаштування ---
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5
MODEL_INPUT_SIZE = (640, 640)  # Ширина, Висота
CLASS_NAMES = ['License_Plate']
TARGET_CLASS_ID = 0
DEBUG_MODE = False  # ОПТИМІЗАЦІЯ: Додано для керування виведенням налагоджувальної інформації


# --- Допоміжні функції ---

def preprocess_frame_onnx_tflite(frame, input_size_wh):
    net_w, net_h = input_size_wh
    img_h, img_w, _ = frame.shape

    # ОПТИМІЗАЦІЯ: Використовуємо cv2.INTER_AREA для зменшення, якщо це доцільно,
    # але для збільшення або змішаного масштабування INTER_LINEAR є хорошим компромісом.
    # Для моделей часто важлива швидкість, тому INTER_LINEAR залишається розумним вибором.
    # Якщо якість падає, можна спробувати cv2.INTER_CUBIC (повільніше).
    interpolation_method = cv2.INTER_LINEAR

    scale = min(net_w / img_w, net_h / img_h)
    new_unpad_w, new_unpad_h = int(round(img_w * scale)), int(round(img_h * scale))
    dw, dh = (net_w - new_unpad_w) / 2, (net_h - new_unpad_h) / 2

    if (img_w, img_h) != (new_unpad_w, new_unpad_h):
        resized_img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=interpolation_method)
    else:
        resized_img = frame  # ОПТИМІЗАЦІЯ: Уникаємо зайвого копіювання, якщо розмір не змінився
        # .copy() буде викликано пізніше в copyMakeBorder, якщо потрібно

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # ОПТИМІЗАЦІЯ: Використовуємо BORDER_REPLICATE або BORDER_CONSTANT.
    # (114, 114, 114) - стандартне значення для YOLO letterbox.
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Нормалізація та створення блобу. swapRB=True, якщо модель очікує RGB, а OpenCV дає BGR.
    blob = cv2.dnn.blobFromImage(padded_img, 1.0 / 255.0, (net_w, net_h), swapRB=True, crop=False)
    return blob, scale, dw, dh


def postprocess_yolo_output(outputs_list, original_shape_hw, input_shape_wh,
                            scale_letterbox, dw_letterbox, dh_letterbox,
                            conf_threshold, nms_threshold, debug_frame_count=0):
    img_h, img_w = original_shape_hw
    net_w, net_h = input_shape_wh
    boxes, confidences, class_ids_list = [], [], []

    if not outputs_list or outputs_list[0] is None or outputs_list[0].size == 0:
        if DEBUG_MODE and debug_frame_count < 2: print(
            f"POSTPROCESS (Кадр {debug_frame_count}): Список виходів порожній або тензор порожній.")
        return [], [], []

    predictions = np.squeeze(outputs_list[0])  # Зазвичай вихід один, тому беремо [0]

    # Перевірка та можливе транспонування вихідного тензора
    # Формат YOLO зазвичай (batch_size, num_detections, 5 + num_classes) або (batch_size, 5 + num_classes, num_detections)
    # Після squeeze, якщо batch_size=1, то (num_detections, 5 + num_classes) або (5 + num_classes, num_detections)
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1] and \
            predictions.shape[0] <= (5 + len(CLASS_NAMES) + 3):  # +3 для гнучкості з масками/точками
        if DEBUG_MODE and debug_frame_count < 2:
            print(f"POSTPROCESS (Кадр {debug_frame_count}): Транспонування вихідного тензора з {predictions.shape}...")
        predictions = predictions.T

    if predictions.ndim != 2 or predictions.shape[0] == 0:
        if DEBUG_MODE and debug_frame_count < 2:
            print(
                f"POSTPROCESS (Кадр {debug_frame_count}): Неочікувана розмірність ({predictions.ndim}) або порожній тензор детекцій ({predictions.shape}) після можливого транспонування.")
        return [], [], []

    num_predictions_total, num_attributes = predictions.shape
    if DEBUG_MODE and debug_frame_count < 2:
        print(
            f"POSTPROCESS (Кадр {debug_frame_count}): Форма предикцій для обробки (N, D): ({num_predictions_total}, {num_attributes})")
        if num_predictions_total > 0:
            print(f"  Зразок перших 2 предикцій (деквантовані FP32):\n{predictions[:2, :]}")
            # Детальніше логування статистики впевненості
            if num_attributes == 5:  # cx, cy, w, h, conf
                conf_column = predictions[:, 4]
                print(
                    f"  Статистика для стовпчика впевненості (атрибут 4): Мін={conf_column.min():.4f}, Макс={conf_column.max():.4f}, Середнє={conf_column.mean():.4f}")
            elif num_attributes >= (5 + len(CLASS_NAMES)):  # cx, cy, w, h, obj_conf, class1_conf, ...
                obj_conf_column = predictions[:, 4]
                # Клас може бути один або декілька
                class_scores_columns = predictions[:, 5: 5 + len(CLASS_NAMES)]
                print(
                    f"  Статистика для стовпчика об'єктності (атрибут 4): Мін={obj_conf_column.min():.4f}, Макс={obj_conf_column.max():.4f}, Середнє={obj_conf_column.mean():.4f}")
                if class_scores_columns.size > 0:
                    # Якщо один цільовий клас, то class_scores_columns[:, TARGET_CLASS_ID]
                    specific_class_conf_column = class_scores_columns[:, TARGET_CLASS_ID] if TARGET_CLASS_ID < \
                                                                                             class_scores_columns.shape[
                                                                                                 1] else class_scores_columns[
                                                                                                         :, 0]
                    print(
                        f"  Статистика для стовпчика класу {CLASS_NAMES[TARGET_CLASS_ID]} (атрибут {5 + TARGET_CLASS_ID}): Мін={specific_class_conf_column.min():.4f}, Макс={specific_class_conf_column.max():.4f}, Середнє={specific_class_conf_column.mean():.4f}")

    detections_passed_conf_filter = 0
    for pred_idx in range(num_predictions_total):
        pred = predictions[pred_idx]
        current_confidence = -1.0
        class_id = -1

        if num_attributes < 4: continue  # Потрібні як мінімум x, y, w, h

        # Координати рамки (центр x, центр y, ширина, висота) - нормалізовані
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

        if num_attributes == (5 + len(CLASS_NAMES)):  # YOLOv8, YOLOv5 (з class scores)
            object_confidence = pred[4]
            if len(CLASS_NAMES) == 1:  # Якщо тільки один клас у моделі
                class_score_for_target_class = pred[5 + TARGET_CLASS_ID] if (
                                                                                        5 + TARGET_CLASS_ID) < num_attributes else 0.0
                current_confidence = object_confidence * class_score_for_target_class
                class_id = TARGET_CLASS_ID
            else:  # Якщо класів багато, але нас цікавить один
                class_scores = pred[5:]  # Всі оцінки класів
                # class_id = np.argmax(class_scores) # Якщо б ми шукали найкращий клас
                # max_class_score = class_scores[class_id]
                # current_confidence = object_confidence * max_class_score
                # Для фіксованого TARGET_CLASS_ID:
                if TARGET_CLASS_ID < len(class_scores):
                    class_score_for_target_class = class_scores[TARGET_CLASS_ID]
                    current_confidence = object_confidence * class_score_for_target_class
                    class_id = TARGET_CLASS_ID
                else:
                    continue  # Немає такого TARGET_CLASS_ID у виході моделі

        elif num_attributes == 5 and len(CLASS_NAMES) == 1:  # Спрощений випадок: x,y,w,h, conf (для одного класу)
            current_confidence = pred[4]
            class_id = TARGET_CLASS_ID
        else:
            if DEBUG_MODE and debug_frame_count < 2 and pred_idx < 5:  # Логуємо тільки перші кілька пропусків
                print(
                    f"  Пропуск детекції {pred_idx} з непідтримуваною кількістю атрибутів {num_attributes} для поточних CLASS_NAMES. pred[0:5]={pred[0:min(5, num_attributes)]}")
            continue

        if DEBUG_MODE and debug_frame_count < 2 and pred_idx < 10:  # Логуємо перші 10 детекцій
            print(
                f"  Детекція {pred_idx}: координати(cx,cy,w,h)=({cx:.2f},{cy:.2f},{w:.2f},{h:.2f}), розрах. впевненість = {current_confidence:.4f} (поріг = {conf_threshold:.4f}), клас_id={class_id}")

        if current_confidence >= conf_threshold and class_id == TARGET_CLASS_ID:  # Перевіряємо і цільовий клас
            detections_passed_conf_filter += 1
            # Перерахунок координат з нормалізованих (0-1 відносно вхідного розміру моделі)
            # назад до оригінальних розмірів кадру, враховуючи letterbox
            box_x_center_on_net = cx * net_w  # cx - це відносний центр на зображенні розміром net_w x net_h
            box_y_center_on_net = cy * net_h
            box_width_on_net = w * net_w
            box_height_on_net = h * net_h

            # Повернення до розмірів зображення до letterbox padding, але все ще масштабованого
            box_x_center_on_resized_original = box_x_center_on_net - dw_letterbox
            box_y_center_on_resized_original = box_y_center_on_net - dh_letterbox

            # Повернення до оригінальних розмірів зображення
            original_box_x_center = box_x_center_on_resized_original / scale_letterbox
            original_box_y_center = box_y_center_on_resized_original / scale_letterbox
            original_box_width = box_width_on_net / scale_letterbox
            original_box_height = box_height_on_net / scale_letterbox

            # Перетворення в (x_min, y_min, width, height)
            x_min = int(original_box_x_center - original_box_width / 2)
            y_min = int(original_box_y_center - original_box_height / 2)

            # Важливо: переконатися, що ширина та висота позитивні
            current_w_int = int(original_box_width)
            current_h_int = int(original_box_height)

            if current_w_int > 0 and current_h_int > 0:
                boxes.append([x_min, y_min, current_w_int, current_h_int])
                confidences.append(float(current_confidence))
                class_ids_list.append(class_id)
            elif DEBUG_MODE and debug_frame_count < 2:
                print(
                    f"    Відфільтровано невалідну рамку: w_orig={original_box_width:.2f} (int: {current_w_int}), h_orig={original_box_height:.2f} (int: {current_h_int}) для детекції {pred_idx}")

    if DEBUG_MODE and debug_frame_count < 2:
        print(
            f"POSTPROCESS (Кадр {debug_frame_count}): Детекцій пройшло поріг впевненості ({conf_threshold}): {detections_passed_conf_filter} / {num_predictions_total}")
        print(f"  Кількість рамок передано в NMS: {len(boxes)}")

    if not boxes: return [], [], []

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_boxes, final_confidences, final_class_ids = [], [], []
    if len(indices) > 0:
        # Якщо indices - це масив numpy вигляду [[idx1], [idx2], ...], то flatten() перетворить його на [idx1, idx2, ...]
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids_list[i])

    if DEBUG_MODE and debug_frame_count < 2:
        print(f"POSTPROCESS (Кадр {debug_frame_count}): Детекцій після NMS: {len(final_boxes)}")
        if len(final_boxes) > 0:
            print(f"  Приклад першої фінальної рамки: box={final_boxes[0]}, conf={final_confidences[0]}")
        elif len(boxes) > 0:  # Були кандидати, але NMS всіх відфільтрував
            print(f"  NMS відфільтрував усі {len(boxes)} рамки.")

    return final_boxes, final_confidences, final_class_ids


def draw_detections(frame, boxes, confidences, class_ids, class_names_list):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]

        if class_id == TARGET_CLASS_ID:  # Малюємо тільки цільовий клас
            color = (0, 255, 0)  # Зелений для номерних знаків
            label = f"{class_names_list[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def main(video_path, model_type, model_path_pt, model_path_onnx, model_path_tflite, num_threads_tflite, frame_skip):
    global DEBUG_MODE  # Дозволяємо змінювати глобальну змінну

    model, onnx_session, tflite_interpreter = None, None, None
    tflite_input_details, tflite_output_details = None, None

    # Встановлення рівня деталізації в залежності від прапорця
    if not args.debug:
        DEBUG_MODE = False
        print("Режим налагодження вимкнено. Для детального логування запустіть з прапорцем --debug.")
    else:
        DEBUG_MODE = True
        print("Режим налагодження увімкнено.")

    if model_type == 'pt':
        if not ULTRALYTICS_AVAILABLE: print(
            "Неможливо завантажити .pt модель: бібліотека 'ultralytics' не встановлена."); return
        if not model_path_pt: print("Шлях до .pt моделі (--pt_path) не вказано."); return
        try:
            model = YOLO(model_path_pt)
            print(f"Успішно завантажено .pt модель: {model_path_pt}")
        except Exception as e:
            print(f"Помилка завантаження .pt моделі: {e}");
            return
    elif model_type == 'onnx':
        if not ONNXRUNTIME_AVAILABLE: print(
            "Неможливо завантажити .onnx модель: бібліотека 'onnxruntime' не встановлена."); return
        if not model_path_onnx: print("Шлях до .onnx моделі (--onnx_path) не вказано."); return
        try:
            # ОПТИМІЗАЦІЯ: Спробуємо додати провайдери, специфічні для ARM, якщо доступні
            # 'CPUExecutionProvider' - базовий, 'ArmComputeLibraryExecutionProvider' (ACL) - може бути швидшим на ARM.
            # Порядок важливий: ONNX Runtime спробує їх по черзі.
            available_providers = onnxruntime.get_available_providers()
            providers_to_try = []
            if 'ArmComputeLibraryExecutionProvider' in available_providers:
                providers_to_try.append('ArmComputeLibraryExecutionProvider')
            providers_to_try.append('CPUExecutionProvider')  # Завжди як запасний варіант

            onnx_session = onnxruntime.InferenceSession(model_path_onnx, providers=providers_to_try)
            print(f"Успішно завантажено .onnx модель: {model_path_onnx} з провайдерами: {onnx_session.get_providers()}")
        except Exception as e:
            print(f"Помилка завантаження .onnx моделі: {e}");
            return
    elif model_type == 'tflite':
        if not TENSORFLOW_AVAILABLE: print(
            "Неможливо завантажити .tflite модель: бібліотека 'tensorflow' не встановлена."); return
        if not model_path_tflite: print("Шлях до .tflite моделі (--tflite_path) не вказано."); return
        try:
            tflite_interpreter = tf.lite.Interpreter(model_path=model_path_tflite, num_threads=num_threads_tflite)
            tflite_interpreter.allocate_tensors()
            tflite_input_details = tflite_interpreter.get_input_details()
            tflite_output_details = tflite_interpreter.get_output_details()
            print(
                f"Успішно завантажено .tflite модель: {model_path_tflite} (використовується {num_threads_tflite} потоків)")
            if DEBUG_MODE:
                print(f"  Вхідні деталі TFLite: {tflite_input_details[0]}")
                print(f"  Вихідні деталі TFLite: {tflite_output_details[0]}")
        except Exception as e:
            print(f"Помилка завантаження .tflite моделі: {e}");
            return
    else:
        print(f"Невідомий тип моделі: {model_type}. Доступні типи: 'pt', 'onnx', 'tflite'.");
        return

    # ОПТИМІЗАЦІЯ: Можна спробувати різні бекенди для cv2.VideoCapture, якщо стандартний працює погано
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) # Наприклад
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Помилка: Неможливо відкрити відеофайл: {video_path}"); return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Інформація про відео: {frame_width}x{frame_height} @ {original_fps:.2f} FPS")

    processed_frame_count, total_inference_time, total_frames_passed_skip = 0, 0, 0
    display_frame_count = 0  # Лічильник для пропуску кадрів

    print(f"\nОбробка відео з моделлю {model_type.upper()}...")

    # Поріг впевненості для TFLite моделей, особливо INT8, може потребувати коригування.
    # Для INT8 моделей вихідні значення можуть бути менш "чистими".
    # Цей поріг передається в postprocess_yolo_output.
    current_conf_threshold_for_postprocess = CONF_THRESHOLD
    if model_type == 'tflite':
        # Можливо, для квантованих TFLite моделей знадобиться нижчий поріг для початкового відбору.
        # Або ж, якщо модель добре квантована, стандартний поріг може бути ОК.
        # Це значення для експериментів.
        # current_conf_threshold_for_postprocess = 0.1 # Наприклад, для дуже "шумних" INT8
        if DEBUG_MODE:
            print(
                f"УВАГА: Для TFLite використовується поріг впевненості для постобробки: {current_conf_threshold_for_postprocess}")
            print(f"       (Стандартний CONF_THRESHOLD = {CONF_THRESHOLD})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        display_frame_count += 1
        if frame_skip > 0 and display_frame_count % (frame_skip + 1) != 1:
            # Пропускаємо цей кадр для обробки, але показуємо його, щоб відео не "рвалося"
            # Якщо потрібно також пропускати показ, то цей блок можна перемістити вище
            # і не викликати cv2.imshow для пропущених кадрів.
            # Однак, для вимірювання FPS моделі важливо обробляти менше кадрів.
            if cv2.getWindowProperty(f"Real-time Detection - {model_type.upper()}", cv2.WND_PROP_VISIBLE) >= 1:
                # Показуємо попередній кадр з детекціями, якщо вікно видиме
                # або просто оригінальний кадр, якщо frame_with_detections ще не було
                if 'frame_with_detections' in locals():
                    cv2.imshow(f"Real-time Detection - {model_type.upper()}", frame_with_detections)
                else:
                    cv2.imshow(f"Real-time Detection - {model_type.upper()}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        total_frames_passed_skip += 1
        start_time_inference = time.perf_counter()  # Більш точний час для вимірювання продуктивності

        original_shape_hw = frame.shape[:2]
        current_boxes, current_confidences, current_class_ids = [], [], []

        if model_type == 'pt' and model:
            # verbose=False вже було, це добре.
            # classes=[TARGET_CLASS_ID] - фільтрація на рівні моделі, якщо підтримується.
            results = model(frame, verbose=False, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, classes=[TARGET_CLASS_ID])
            # Обробка результатів для Ultralytics YOLO
            for res_obj in results:  # Зазвичай один об'єкт результату для одного зображення
                for box_obj in res_obj.boxes:  # boxes - це атрибут з інформацією про рамки
                    xyxy, conf, cls_id = box_obj.xyxy[0].cpu().numpy(), \
                        float(box_obj.conf[0].cpu().numpy()), \
                        int(box_obj.cls[0].cpu().numpy())
                    # Перевірка, чи клас відповідає цільовому (хоча вже фільтрували)
                    if cls_id == TARGET_CLASS_ID:
                        x1, y1, x2, y2 = map(int, xyxy)
                        current_boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
                        current_confidences.append(conf)
                        current_class_ids.append(cls_id)

        elif model_type == 'onnx' and onnx_session:
            input_name = onnx_session.get_inputs()[0].name
            blob, scale_l, dw_l, dh_l = preprocess_frame_onnx_tflite(frame, MODEL_INPUT_SIZE)
            onnx_outputs = onnx_session.run(None, {input_name: blob})
            current_boxes, current_confidences, current_class_ids = postprocess_yolo_output(
                onnx_outputs, original_shape_hw, MODEL_INPUT_SIZE, scale_l, dw_l, dh_l,
                CONF_THRESHOLD, NMS_THRESHOLD, debug_frame_count=total_frames_passed_skip if DEBUG_MODE else -1
            )

        elif model_type == 'tflite' and tflite_interpreter:
            input_details_single = tflite_input_details[0]
            is_int8_input = input_details_single['dtype'] == np.int8

            blob_fp32, scale_l, dw_l, dh_l = preprocess_frame_onnx_tflite(frame, MODEL_INPUT_SIZE)
            input_data_to_set = None

            if is_int8_input:
                input_quant_params = input_details_single.get('quantization_parameters', {})
                # .get('scales', [1.0])[0] - безпечне отримання, якщо параметр відсутній
                input_scale_val = input_quant_params.get('scales', [1.0])[0]
                input_zero_point_val = input_quant_params.get('zero_points', [0])[0]

                if DEBUG_MODE and total_frames_passed_skip < 2: print(
                    f"TFLite вхід INT8. Квант. параметри: scale={input_scale_val:.8f}, zero_point={input_zero_point_val}")

                if abs(input_scale_val) < 1e-9:  # Запобігання діленню на нуль
                    print(
                        "ПОМИЛКА: Вхідний масштаб INT8 TFLite близький до нуля! Використовується FP32 як запасний варіант.")
                    # Це може не спрацювати, якщо модель суворо очікує INT8, але це краще, ніж креш.
                    input_data_to_set = blob_fp32.astype(
                        input_details_single['dtype'])  # Спроба конвертувати у тип, який очікує модель
                else:
                    input_data_to_set = np.clip((blob_fp32 / input_scale_val) + input_zero_point_val, -128, 127).astype(
                        np.int8)
            else:  # FP32 input
                if DEBUG_MODE and total_frames_passed_skip < 2: print("TFLite вхід FLOAT32.")
                input_data_to_set = blob_fp32.astype(input_details_single['dtype'])  # Зазвичай np.float32

            tflite_interpreter.set_tensor(input_details_single['index'], input_data_to_set)
            tflite_interpreter.invoke()

            if DEBUG_MODE and total_frames_passed_skip < 2: print(
                f"\n--- TFLite Output Debug Info (Кадр {total_frames_passed_skip}) ---"); print(
                f"К-ть вих. тензорів: {len(tflite_output_details)}")

            output_data_list_fp32 = []
            for i in range(len(tflite_output_details)):
                output_details_single = tflite_output_details[i]
                output_tensor_raw = tflite_interpreter.get_tensor(output_details_single['index'])
                dtype, shape, name = output_details_single['dtype'], output_details_single['shape'], \
                output_details_single['name']
                quant_params = output_details_single.get('quantization_parameters', {})

                if DEBUG_MODE and total_frames_passed_skip < 2: print(
                    f"  Вих. тензор {i}: name='{name}', shape={shape}, dtype={dtype}, quant_params={quant_params}")

                current_output_fp32 = None
                if dtype == np.int8:  # Якщо вихід квантований
                    output_scale_val = quant_params.get('scales', [1.0])[0]
                    output_zero_point_val = quant_params.get('zero_points', [0])[0]
                    if DEBUG_MODE and total_frames_passed_skip < 2:
                        print(
                            f"    Деквант. INT8 вих. тензор {i} з scale={output_scale_val:.8f}, zero_point={output_zero_point_val}")
                        if output_tensor_raw.size > 0 and len(output_tensor_raw.shape) > 2 and output_tensor_raw.shape[
                            1] == (5 + len(CLASS_NAMES)) and output_tensor_raw.shape[0] == 1:
                            # Приклад: (1, 6, 8400) -> (1, num_attrs, num_preds)
                            # Атрибут об'єктності (індекс 4)
                            print(
                                f"      Сирі INT8 значення для атрибута об'єктності (перші 10 з {output_tensor_raw.shape[2]}): {output_tensor_raw[0, 4, :10]}")

                    if abs(output_scale_val) < 1e-9:
                        print(
                            f"    ПОМИЛКА: Вих. масштаб INT8 тензора {i} ('{name}') близький до нуля! Використовується сирий INT8 як FP32.")
                        current_output_fp32 = output_tensor_raw.astype(np.float32)
                    else:
                        current_output_fp32 = (output_tensor_raw.astype(
                            np.float32) - output_zero_point_val) * output_scale_val
                else:  # Якщо вихід FP32
                    current_output_fp32 = output_tensor_raw.astype(np.float32)  # Забезпечуємо, що це float32

                output_data_list_fp32.append(current_output_fp32)

            if DEBUG_MODE and total_frames_passed_skip < 2: print("--- Кінець TFLite Output Debug Info ---\n")

            # Припускаємо, що перший вихідний тензор є основним для детекцій
            primary_output_fp32_for_postprocess = output_data_list_fp32[0] if output_data_list_fp32 and \
                                                                              output_data_list_fp32[
                                                                                  0] is not None else np.array([])

            if DEBUG_MODE and total_frames_passed_skip < 2 and primary_output_fp32_for_postprocess.size > 0: print(
                f"Вих. тензор '{tflite_output_details[0]['name']}' форми {primary_output_fp32_for_postprocess.shape} передається в постобробку.")

            current_boxes, current_confidences, current_class_ids = postprocess_yolo_output(
                [primary_output_fp32_for_postprocess], original_shape_hw, MODEL_INPUT_SIZE,
                scale_l, dw_l, dh_l, current_conf_threshold_for_postprocess, NMS_THRESHOLD,
                # Використовуємо налаштований поріг
                debug_frame_count=total_frames_passed_skip if DEBUG_MODE else -1
            )

        end_time_inference = time.perf_counter()
        elapsed_time_inference = end_time_inference - start_time_inference

        # Рахуємо FPS тільки для оброблених кадрів
        if elapsed_time_inference > 0:
            fps_current_inference = 1 / elapsed_time_inference
            if total_frames_passed_skip > 1:  # Не враховуємо перший кадр для середнього FPS інференсу
                total_inference_time += elapsed_time_inference
                processed_frame_count += 1
        else:
            fps_current_inference = 0

        # Малювання детекцій
        frame_with_detections = draw_detections(frame.copy(), current_boxes, current_confidences, current_class_ids,
                                                CLASS_NAMES)

        # Відображення FPS на кадрі
        avg_fps_inference_so_far = (processed_frame_count / total_inference_time) if total_inference_time > 0 else 0
        cv2.putText(frame_with_detections,
                    f"FPS (inf): {fps_current_inference:.2f} (avg: {avg_fps_inference_so_far:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_with_detections, f"Detections: {len(current_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)

        cv2.imshow(f"Real-time Detection - {model_type.upper()}", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Розрахунок середнього FPS інференсу (без урахування першого кадру, якщо їх було більше одного)
    avg_fps_inference = (processed_frame_count / total_inference_time) if total_inference_time > 0 else 0

    print(f"\nОбробку завершено.")
    print(f"Всього кадрів у відео оброблено (з урахуванням пропуску): {total_frames_passed_skip}")
    if processed_frame_count > 0:
        print(
            f"Середній FPS інференсу (без першого кадру, якщо >1): {avg_fps_inference:.2f} для моделі {model_type.upper()}")
    elif total_frames_passed_skip == 1 and total_inference_time > 0:  # Якщо оброблено лише один кадр
        avg_fps_inference = 1 / total_inference_time
        print(f"FPS інференсу для єдиного обробленого кадру: {avg_fps_inference:.2f} для моделі {model_type.upper()}")
    else:
        print(f"Недостатньо кадрів оброблено для розрахунку середнього FPS інференсу.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Тестування моделей YOLO (.pt, .onnx, .tflite) на відео для Raspberry Pi.")
    parser.add_argument("--video", type=str, required=True, help="Шлях до відеофайлу.")
    parser.add_argument("--model_type", type=str, required=True, choices=['pt', 'onnx', 'tflite'],
                        help="Тип моделі для тестування (pt, onnx, tflite).")
    parser.add_argument("--pt_path", type=str, default=None, help="Шлях до .pt моделі (Ultralytics YOLO).")
    parser.add_argument("--onnx_path", type=str, default=None, help="Шлях до .onnx моделі.")
    parser.add_argument("--tflite_path", type=str, default=None, help="Шлях до .tflite моделі.")

    # ОПТИМІЗАЦІЯ: Додано аргументи для Raspberry Pi
    parser.add_argument("--tflite_threads", type=int, default=4,
                        help="Кількість потоків для TensorFlow Lite Interpreter (рекомендовано 4 для RPi 4/5).")
    parser.add_argument("--frame_skip", type=int, default=0,
                        help="Кількість кадрів, які потрібно пропускати між обробками (0 - не пропускати).")
    parser.add_argument("--debug", action='store_true', help="Увімкнути детальне логування для налагодження.")

    args = parser.parse_args()

    if not (ULTRALYTICS_AVAILABLE or ONNXRUNTIME_AVAILABLE or TENSORFLOW_AVAILABLE):
        print("ПОМИЛКА: Жодна з необхідних бібліотек для інференсу моделей не встановлена.")
        print("Будь ласка, встановіть потрібні бібліотеки, наприклад:")
        print("  pip install ultralytics onnxruntime tensorflow opencv-python numpy")
    else:
        main(args.video, args.model_type, args.pt_path, args.onnx_path, args.tflite_path, args.tflite_threads,
             args.frame_skip)