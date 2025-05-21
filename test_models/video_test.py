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
DEBUG_MODE = False


# --- Допоміжні функції ---

def preprocess_frame_onnx_tflite(frame, input_size_wh):
    net_w, net_h = input_size_wh
    img_h, img_w, _ = frame.shape
    interpolation_method = cv2.INTER_LINEAR
    scale = min(net_w / img_w, net_h / img_h)
    new_unpad_w, new_unpad_h = int(round(img_w * scale)), int(round(img_h * scale))
    dw, dh = (net_w - new_unpad_w) / 2, (net_h - new_unpad_h) / 2

    if (img_w, img_h) != (new_unpad_w, new_unpad_h):
        resized_img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=interpolation_method)
    else:
        resized_img = frame
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
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

    predictions = np.squeeze(outputs_list[0])
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1] and \
            predictions.shape[0] <= (5 + len(CLASS_NAMES) + 3):
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
            if num_attributes == 5:
                conf_column = predictions[:, 4]
                print(
                    f"  Статистика для стовпчика впевненості (атрибут 4): Мін={conf_column.min():.4f}, Макс={conf_column.max():.4f}, Середнє={conf_column.mean():.4f}")
            elif num_attributes >= (5 + len(CLASS_NAMES)):
                obj_conf_column = predictions[:, 4]
                class_scores_columns = predictions[:, 5: 5 + len(CLASS_NAMES)]
                print(
                    f"  Статистика для стовпчика об'єктності (атрибут 4): Мін={obj_conf_column.min():.4f}, Макс={obj_conf_column.max():.4f}, Середнє={obj_conf_column.mean():.4f}")
                if class_scores_columns.size > 0:
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
        if num_attributes < 4: continue
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

        if num_attributes == (5 + len(CLASS_NAMES)):
            object_confidence = pred[4]
            if len(CLASS_NAMES) == 1:
                class_score_for_target_class = pred[5 + TARGET_CLASS_ID] if (
                                                                                        5 + TARGET_CLASS_ID) < num_attributes else 0.0
                current_confidence = object_confidence * class_score_for_target_class
                class_id = TARGET_CLASS_ID
            else:
                if TARGET_CLASS_ID < len(pred[5:]):
                    class_score_for_target_class = pred[5 + TARGET_CLASS_ID]
                    current_confidence = object_confidence * class_score_for_target_class
                    class_id = TARGET_CLASS_ID
                else:
                    continue
        elif num_attributes == 5 and len(CLASS_NAMES) == 1:
            current_confidence = pred[4]
            class_id = TARGET_CLASS_ID
        else:
            if DEBUG_MODE and debug_frame_count < 2 and pred_idx < 5:
                print(
                    f"  Пропуск детекції {pred_idx} з непідтримуваною кількістю атрибутів {num_attributes} для поточних CLASS_NAMES. pred[0:5]={pred[0:min(5, num_attributes)]}")
            continue

        if DEBUG_MODE and debug_frame_count < 2 and pred_idx < 10:
            print(
                f"  Детекція {pred_idx}: координати(cx,cy,w,h)=({cx:.2f},{cy:.2f},{w:.2f},{h:.2f}), розрах. впевненість = {current_confidence:.4f} (поріг = {conf_threshold:.4f}), клас_id={class_id}")

        if current_confidence >= conf_threshold and class_id == TARGET_CLASS_ID:
            detections_passed_conf_filter += 1
            box_x_center_on_net = cx * net_w
            box_y_center_on_net = cy * net_h
            box_width_on_net = w * net_w
            box_height_on_net = h * net_h
            box_x_center_on_resized_original = box_x_center_on_net - dw_letterbox
            box_y_center_on_resized_original = box_y_center_on_net - dh_letterbox
            original_box_x_center = box_x_center_on_resized_original / scale_letterbox
            original_box_y_center = box_y_center_on_resized_original / scale_letterbox
            original_box_width = box_width_on_net / scale_letterbox
            original_box_height = box_height_on_net / scale_letterbox
            x_min = int(original_box_x_center - original_box_width / 2)
            y_min = int(original_box_y_center - original_box_height / 2)
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    final_boxes, final_confidences, final_class_ids = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids_list[i])

    if DEBUG_MODE and debug_frame_count < 2:
        print(f"POSTPROCESS (Кадр {debug_frame_count}): Детекцій після NMS: {len(final_boxes)}")
        if len(final_boxes) > 0:
            print(f"  Приклад першої фінальної рамки: box={final_boxes[0]}, conf={final_confidences[0]}")
        elif len(boxes) > 0:
            print(f"  NMS відфільтрував усі {len(boxes)} рамки.")
    return final_boxes, final_confidences, final_class_ids


def draw_detections(frame, boxes, confidences, class_ids, class_names_list):
    # Ця функція все ще потрібна, навіть якщо відображення вимкнено,
    # оскільки вона готує `frame_with_detections`.
    # Якщо б малювання було дуже ресурсомістким, можна було б і його зробити умовним.
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        if class_id == TARGET_CLASS_ID:
            color = (0, 255, 0)
            label = f"{class_names_list[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def main(video_path, model_type, model_path_pt, model_path_onnx, model_path_tflite, num_threads_tflite, frame_skip,
         no_display_flag):  # Додано no_display_flag
    global DEBUG_MODE

    if not args.debug:  # args доступний глобально, якщо main викликається з if __name__ == "__main__":
        DEBUG_MODE = False
        if not no_display_flag:  # Не друкувати, якщо дисплей вимкнено і це може бути headless
            print("Режим налагодження вимкнено. Для детального логування запустіть з прапорцем --debug.")
    else:
        DEBUG_MODE = True
        print("Режим налагодження увімкнено.")

    model, onnx_session, tflite_interpreter = None, None, None
    tflite_input_details, tflite_output_details = None, None

    if model_type == 'pt':
        if not ULTRALYTICS_AVAILABLE: print(
            "Неможливо завантажити .pt модель: бібліотека 'ultralytics' не встановлена."); return
        if not model_path_pt: print("Шлях до .pt моделі (--pt_path) не вказано."); return
        try:
            model = YOLO(model_path_pt); print(f"Успішно завантажено .pt модель: {model_path_pt}")
        except Exception as e:
            print(f"Помилка завантаження .pt моделі: {e}"); return
    elif model_type == 'onnx':
        if not ONNXRUNTIME_AVAILABLE: print(
            "Неможливо завантажити .onnx модель: бібліотека 'onnxruntime' не встановлена."); return
        if not model_path_onnx: print("Шлях до .onnx моделі (--onnx_path) не вказано."); return
        try:
            available_providers = onnxruntime.get_available_providers()
            providers_to_try = []
            if 'ArmComputeLibraryExecutionProvider' in available_providers: providers_to_try.append(
                'ArmComputeLibraryExecutionProvider')
            providers_to_try.append('CPUExecutionProvider')
            onnx_session = onnxruntime.InferenceSession(model_path_onnx, providers=providers_to_try)
            print(f"Успішно завантажено .onnx модель: {model_path_onnx} з провайдерами: {onnx_session.get_providers()}")
        except Exception as e:
            print(f"Помилка завантаження .onnx моделі: {e}"); return
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
            print(f"Помилка завантаження .tflite моделі: {e}"); return
    else:
        print(f"Невідомий тип моделі: {model_type}. Доступні типи: 'pt', 'onnx', 'tflite'.");
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Помилка: Неможливо відкрити відеофайл: {video_path}"); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps_video = cap.get(cv2.CAP_PROP_FPS)  # Перейменовано, щоб уникнути конфлікту
    print(f"Інформація про відео: {frame_width}x{frame_height} @ {original_fps_video:.2f} FPS")

    processed_frame_count, total_inference_time, total_frames_passed_skip = 0, 0, 0
    loop_frame_counter = 0  # Лічильник кадрів циклу для frame_skip

    # Створюємо вікно заздалегідь, якщо відображення увімкнено
    window_name = f"Real-time Detection - {model_type.upper()}"
    if not no_display_flag:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # Або cv2.WINDOW_NORMAL для зміни розміру

    print(f"\nОбробка відео з моделлю {model_type.upper()}...")
    current_conf_threshold_for_postprocess = CONF_THRESHOLD
    if model_type == 'tflite' and DEBUG_MODE:
        print(
            f"УВАГА: Для TFLite використовується поріг впевненості для постобробки: {current_conf_threshold_for_postprocess}")

    # Змінна для збереження останнього кадру з детекціями для показу під час пропуску кадрів
    last_frame_with_detections = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        loop_frame_counter += 1
        perform_processing_this_frame = True
        if frame_skip > 0 and (loop_frame_counter - 1) % (
                frame_skip + 1) != 0:  # (loop_frame_counter -1) щоб перший кадр (1) оброблявся
            perform_processing_this_frame = False

        if perform_processing_this_frame:
            total_frames_passed_skip += 1
            start_time_inference = time.perf_counter()

            original_shape_hw = frame.shape[:2]
            current_boxes, current_confidences, current_class_ids = [], [], []

            if model_type == 'pt' and model:
                results = model(frame, verbose=False, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, classes=[TARGET_CLASS_ID])
                for res_obj in results:
                    for box_obj in res_obj.boxes:
                        xyxy, conf, cls_id = box_obj.xyxy[0].cpu().numpy(), float(box_obj.conf[0].cpu().numpy()), int(
                            box_obj.cls[0].cpu().numpy())
                        if cls_id == TARGET_CLASS_ID:
                            x1, y1, x2, y2 = map(int, xyxy)
                            current_boxes.append([x1, y1, x2 - x1, y2 - y1]);
                            current_confidences.append(conf);
                            current_class_ids.append(cls_id)
            elif model_type == 'onnx' and onnx_session:
                input_name = onnx_session.get_inputs()[0].name
                blob, scale_l, dw_l, dh_l = preprocess_frame_onnx_tflite(frame, MODEL_INPUT_SIZE)
                onnx_outputs = onnx_session.run(None, {input_name: blob})
                current_boxes, current_confidences, current_class_ids = postprocess_yolo_output(
                    onnx_outputs, original_shape_hw, MODEL_INPUT_SIZE, scale_l, dw_l, dh_l,
                    CONF_THRESHOLD, NMS_THRESHOLD, debug_frame_count=total_frames_passed_skip if DEBUG_MODE else -1)
            elif model_type == 'tflite' and tflite_interpreter:
                input_details_single = tflite_input_details[0]
                is_int8_input = input_details_single['dtype'] == np.int8
                blob_fp32, scale_l, dw_l, dh_l = preprocess_frame_onnx_tflite(frame, MODEL_INPUT_SIZE)
                input_data_to_set = None
                if is_int8_input:
                    input_quant_params = input_details_single.get('quantization_parameters', {})
                    input_scale_val = input_quant_params.get('scales', [1.0])[0]
                    input_zero_point_val = input_quant_params.get('zero_points', [0])[0]
                    if DEBUG_MODE and total_frames_passed_skip < 2: print(
                        f"TFLite вхід INT8. Квант. параметри: scale={input_scale_val:.8f}, zero_point={input_zero_point_val}")
                    if abs(input_scale_val) < 1e-9:
                        print("ПОМИЛКА: Вхідний масштаб INT8 TFLite близький до нуля!");
                        input_data_to_set = blob_fp32.astype(input_details_single['dtype'])
                    else:
                        input_data_to_set = np.clip((blob_fp32 / input_scale_val) + input_zero_point_val, -128,
                                                    127).astype(np.int8)
                else:
                    if DEBUG_MODE and total_frames_passed_skip < 2: print("TFLite вхід FLOAT32.")
                    input_data_to_set = blob_fp32.astype(input_details_single['dtype'])

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
                    if dtype == np.int8:
                        output_scale_val = quant_params.get('scales', [1.0])[0];
                        output_zero_point_val = quant_params.get('zero_points', [0])[0]
                        if DEBUG_MODE and total_frames_passed_skip < 2:
                            print(
                                f"    Деквант. INT8 вих. тензор {i} з scale={output_scale_val:.8f}, zero_point={output_zero_point_val}")
                            if output_tensor_raw.size > 0 and len(output_tensor_raw.shape) > 2 and \
                                    output_tensor_raw.shape[0] == 1 and output_tensor_raw.shape[1] >= 5:
                                print(
                                    f"      Сирі INT8 значення для атрибута об'єктності (перші 10 з {output_tensor_raw.shape[-1]}): {output_tensor_raw[0, 4, :10]}")  # Змінено індекс для останньої розмірності
                        if abs(output_scale_val) < 1e-9:
                            print(f"    ПОМИЛКА: Вих. масштаб INT8 тензора {i} ('{name}') близький до нуля!");
                            current_output_fp32 = output_tensor_raw.astype(np.float32)
                        else:
                            current_output_fp32 = (output_tensor_raw.astype(
                                np.float32) - output_zero_point_val) * output_scale_val
                    else:
                        current_output_fp32 = output_tensor_raw.astype(np.float32)
                    output_data_list_fp32.append(current_output_fp32)
                if DEBUG_MODE and total_frames_passed_skip < 2: print("--- Кінець TFLite Output Debug Info ---\n")
                primary_output_fp32_for_postprocess = output_data_list_fp32[0] if output_data_list_fp32 and \
                                                                                  output_data_list_fp32[
                                                                                      0] is not None else np.array([])
                if DEBUG_MODE and total_frames_passed_skip < 2 and primary_output_fp32_for_postprocess.size > 0: print(
                    f"Вих. тензор '{tflite_output_details[0]['name']}' форми {primary_output_fp32_for_postprocess.shape} передається в постобробку.")
                current_boxes, current_confidences, current_class_ids = postprocess_yolo_output(
                    [primary_output_fp32_for_postprocess], original_shape_hw, MODEL_INPUT_SIZE,
                    scale_l, dw_l, dh_l, current_conf_threshold_for_postprocess, NMS_THRESHOLD,
                    debug_frame_count=total_frames_passed_skip if DEBUG_MODE else -1)

            end_time_inference = time.perf_counter()
            elapsed_time_inference = end_time_inference - start_time_inference

            if elapsed_time_inference > 0:
                fps_current_inference = 1 / elapsed_time_inference
                if total_frames_passed_skip > 1:
                    total_inference_time += elapsed_time_inference
                    processed_frame_count += 1
            else:
                fps_current_inference = 0

            # Малювання детекцій на копії кадру
            frame_to_display = frame.copy()  # Завжди копіюємо перед малюванням
            frame_to_display = draw_detections(frame_to_display, current_boxes, current_confidences, current_class_ids,
                                               CLASS_NAMES)

            avg_fps_inference_so_far = (processed_frame_count / total_inference_time) if total_inference_time > 0 else 0
            cv2.putText(frame_to_display,
                        f"FPS (inf): {fps_current_inference:.2f} (avg: {avg_fps_inference_so_far:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_to_display, f"Detections: {len(current_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            last_frame_with_detections = frame_to_display  # Зберігаємо кадр з детекціями

        # Блок відображення
        if not no_display_flag:
            display_content = None
            if perform_processing_this_frame:  # Якщо кадр оброблено, показуємо його
                display_content = last_frame_with_detections
            elif last_frame_with_detections is not None:  # Якщо кадр пропущено, але є попередній з детекціями
                display_content = last_frame_with_detections
            else:  # Якщо кадр пропущено і це перший кадр (ще немає `last_frame_with_detections`)
                display_content = frame  # Показуємо оригінальний кадр без детекцій

            if display_content is not None:
                cv2.imshow(window_name, display_content)

            # Обробка натискання клавіш (тільки якщо є відображення)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Натиснуто 'q', вихід...")
                break
            elif key == ord('p') or key == ord(' '):  # Пауза на 'p' або пробіл
                print("Пауза. Натисніть будь-яку клавішу для продовження (окрім 'q' для виходу).")
                while True:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key == ord('q'):
                        print("Натиснуто 'q' під час паузи, вихід...")
                        cap.release()
                        cv2.destroyAllWindows()
                        return  # Вихід з main
                    if pause_key is not None:  # Будь-яка інша клавіша
                        break
        else:  # Якщо no_display_flag == True
            if loop_frame_counter % 100 == 0:  # Періодичний вивід в консоль, щоб знати, що скрипт працює
                avg_fps_so_far = (processed_frame_count / total_inference_time) if total_inference_time > 0 else 0
                print(
                    f"Оброблено кадрів (з урахуванням пропуску): {total_frames_passed_skip}, Поточний середній FPS інференсу: {avg_fps_so_far:.2f}")

    avg_fps_inference = (processed_frame_count / total_inference_time) if total_inference_time > 0 else 0
    print(f"\nОбробку завершено.")
    print(
        f"Всього унікальних кадрів з відео оброблено (з урахуванням пропуску --frame_skip): {total_frames_passed_skip}")
    if processed_frame_count > 0:
        print(
            f"Середній FPS інференсу (без першого кадру, якщо >1): {avg_fps_inference:.2f} для моделі {model_type.upper()}")
    elif total_frames_passed_skip == 1 and total_inference_time > 0:
        avg_fps_inference = 1 / total_inference_time  # Немає total_inference_time, тут помилка, має бути elapsed_time_inference
        # Правильніше буде взяти останній fps_current_inference, якщо оброблено лише один кадр
        if 'fps_current_inference' in locals() and total_frames_passed_skip == 1:
            print(
                f"FPS інференсу для єдиного обробленого кадру: {fps_current_inference:.2f} для моделі {model_type.upper()}")
        else:  # Якщо щось пішло не так з fps_current_inference
            print(
                f"FPS інференсу для єдиного обробленого кадру (розрахунковий): {(1 / elapsed_time_inference if 'elapsed_time_inference' in locals() and elapsed_time_inference > 0 else 'N/A'):.2f} для моделі {model_type.upper()}")

    else:
        print(f"Недостатньо кадрів оброблено для розрахунку середнього FPS інференсу.")

    cap.release()
    if not no_display_flag:  # Закриваємо вікна тільки якщо вони були створені
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
    parser.add_argument("--tflite_threads", type=int, default=4,
                        help="Кількість потоків для TensorFlow Lite Interpreter (рекомендовано 4 для RPi 4/5).")
    parser.add_argument("--frame_skip", type=int, default=0,
                        help="Кількість кадрів, які потрібно пропускати між обробками (0 - не пропускати). N=0 обробляє кожен кадр, N=1 обробляє 1, пропускає 1, обробляє 1 ...).")
    parser.add_argument("--debug", action='store_true', help="Увімкнути детальне логування для налагодження.")
    # ОПТИМІЗАЦІЯ: Додано аргумент для вимкнення відображення
    parser.add_argument("--no_display", action='store_true',
                        help="Вимкнути відображення відеовікна (для тестів продуктивності або headless режиму).")

    args = parser.parse_args()

    if not (ULTRALYTICS_AVAILABLE or ONNXRUNTIME_AVAILABLE or TENSORFLOW_AVAILABLE):
        print("ПОМИЛКА: Жодна з необхідних бібліотек для інференсу моделей не встановлена.")
        print(
            "Будь ласка, встановіть потрібні бібліотеки, наприклад:\n  pip install ultralytics onnxruntime tensorflow opencv-python numpy")
    else:
        main(args.video, args.model_type, args.pt_path, args.onnx_path, args.tflite_path, args.tflite_threads,
             args.frame_skip, args.no_display)