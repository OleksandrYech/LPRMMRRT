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
MODEL_INPUT_SIZE = (640, 640)
CLASS_NAMES = ['License_Plate']
TARGET_CLASS_ID = 0


# --- Допоміжні функції ---

def preprocess_frame_onnx_tflite(frame, input_size_wh):
    net_w, net_h = input_size_wh
    img_h, img_w, _ = frame.shape
    scale = min(net_w / img_w, net_h / img_h)
    new_unpad_w, new_unpad_h = int(round(img_w * scale)), int(round(img_h * scale))
    dw, dh = (net_w - new_unpad_w) / 2, (net_h - new_unpad_h) / 2
    if (img_w, img_h) != (new_unpad_w, new_unpad_h):
        resized_img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = frame.copy()
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
        if debug_frame_count < 2: print(
            f"POSTPROCESS (Кадр {debug_frame_count}): Список виходів порожній або тензор порожній.")
        return [], [], []

    predictions = np.squeeze(outputs_list[0])

    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1] and \
            predictions.shape[0] <= (5 + len(CLASS_NAMES) + 3):
        if debug_frame_count < 2:
            print(f"POSTPROCESS (Кадр {debug_frame_count}): Транспонування вихідного тензора з {predictions.shape}...")
        predictions = predictions.T

    if predictions.ndim != 2 or predictions.shape[0] == 0:  # Додано перевірку на порожній predictions після T
        if debug_frame_count < 2:
            print(
                f"POSTPROCESS (Кадр {debug_frame_count}): Неочікувана розмірність ({predictions.ndim}) або порожній тензор детекцій ({predictions.shape})")
        return [], [], []

    num_predictions_total, num_attributes = predictions.shape
    if debug_frame_count < 2:
        print(
            f"POSTPROCESS (Кадр {debug_frame_count}): Форма предикцій для обробки (N, D): ({num_predictions_total}, {num_attributes})")
        if num_predictions_total > 0:
            print(f"  Зразок перших 2 предикцій (деквантовані FP32):\n{predictions[:2, :]}")
            if num_attributes == 5:
                conf_column = predictions[:, 4]
                print(
                    f"  Статистика для стовпчика впевненості (атрибут 4): Мін={conf_column.min():.4f}, Макс={conf_column.max():.4f}, Середнє={conf_column.mean():.4f}")
            elif num_attributes == (5 + len(CLASS_NAMES)):
                obj_conf_column = predictions[:, 4]
                cls_conf_column = predictions[:, 5]
                print(
                    f"  Статистика для стовпчика об'єктності (атрибут 4): Мін={obj_conf_column.min():.4f}, Макс={obj_conf_column.max():.4f}, Середнє={obj_conf_column.mean():.4f}")
                if cls_conf_column.size > 0:
                    print(
                        f"  Статистика для стовпчика класу (атрибут 5): Мін={cls_conf_column.min():.4f}, Макс={cls_conf_column.max():.4f}, Середнє={cls_conf_column.mean():.4f}")

    detections_passed_conf_filter = 0
    for pred_idx in range(num_predictions_total):
        pred = predictions[pred_idx]
        current_confidence = -1.0
        class_id = -1
        if num_attributes < 4: continue
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

        if num_attributes == (5 + len(CLASS_NAMES)):
            object_confidence = pred[4]
            class_scores = pred[5:]
            if not class_scores.size: continue
            class_id = TARGET_CLASS_ID
            max_class_score = class_scores[0] if class_scores.size > 0 else 0.0
            current_confidence = object_confidence * max_class_score
        elif num_attributes == 5 and len(CLASS_NAMES) == 1:
            current_confidence = pred[4]
            class_id = TARGET_CLASS_ID
        else:
            if debug_frame_count < 2 and pred_idx < 5:
                print(
                    f"  Пропуск детекції {pred_idx} з атрибутами {num_attributes}. pred[0:5]={pred[0:min(5, num_attributes)]}")
            continue

        if debug_frame_count < 2 and pred_idx < 10:
            print(
                f"  Детекція {pred_idx}: координати(cx,cy,w,h)=({cx:.2f},{cy:.2f},{w:.2f},{h:.2f}), розрах. впевненість = {current_confidence:.4f} (поріг = {conf_threshold:.4f})")

        if current_confidence >= conf_threshold:
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
            if int(original_box_width) > 0 and int(original_box_height) > 0:
                boxes.append([x_min, y_min, int(original_box_width), int(original_box_height)])
                confidences.append(float(current_confidence))
                class_ids_list.append(class_id)
            elif debug_frame_count < 2:
                print(
                    f"    Відфільтровано невалідну рамку: w={original_box_width:.2f}, h={original_box_height:.2f} для детекції {pred_idx}")

    if debug_frame_count < 2:
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
    if debug_frame_count < 2:
        print(f"POSTPROCESS (Кадр {debug_frame_count}): Детекцій після NMS: {len(final_boxes)}")
        if len(final_boxes) > 0:
            print(f"  Приклад першої фінальної рамки: box={final_boxes[0]}, conf={final_confidences[0]}")
        elif len(boxes) > 0:
            print(f"  NMS відфільтрував усі {len(boxes)} рамки.")
    return final_boxes, final_confidences, final_class_ids


def draw_detections(frame, boxes, confidences, class_ids, class_names_list):
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


def main(video_path, model_type, model_path_pt, model_path_onnx, model_path_tflite):
    model, onnx_session, tflite_interpreter = None, None, None
    tflite_input_details, tflite_output_details = None, None

    if model_type == 'pt':
        if not ULTRALYTICS_AVAILABLE: print("Неможливо: 'ultralytics' не встановлена."); return
        if not model_path_pt: print("Шлях --pt_path не вказано."); return
        try:
            model = YOLO(model_path_pt); print(f".pt модель: {model_path_pt}")
        except Exception as e:
            print(f"Помилка .pt: {e}"); return
    elif model_type == 'onnx':
        if not ONNXRUNTIME_AVAILABLE: print("Неможливо: 'onnxruntime' не встановлена."); return
        if not model_path_onnx: print("Шлях --onnx_path не вказано."); return
        try:
            onnx_session = onnxruntime.InferenceSession(model_path_onnx, providers=['CPUExecutionProvider']); print(
                f".onnx модель: {model_path_onnx}")
        except Exception as e:
            print(f"Помилка .onnx: {e}"); return
    elif model_type == 'tflite':
        if not TENSORFLOW_AVAILABLE: print("Неможливо: 'tensorflow' не встановлена."); return
        if not model_path_tflite: print("Шлях --tflite_path не вказано."); return
        try:
            tflite_interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
            tflite_interpreter.allocate_tensors()
            tflite_input_details = tflite_interpreter.get_input_details()
            tflite_output_details = tflite_interpreter.get_output_details()
            print(f".tflite модель: {model_path_tflite}")
        except Exception as e:
            print(f"Помилка .tflite: {e}"); return
    else:
        print(f"Невідомий тип моделі: {model_type}"); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Помилка: Неможливо відкрити відео: {video_path}"); return
    frame_count, total_fps = 0, 0
    print(f"\nОбробка відео з моделлю {model_type.upper()}...")
    current_conf_threshold_for_postprocess = CONF_THRESHOLD
    if model_type == 'tflite':
        current_conf_threshold_for_postprocess = 0.05
        print(f"УВАГА: Для TFLite встановлено тестовий поріг впевненості: {current_conf_threshold_for_postprocess}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        start_time = time.time()
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
                CONF_THRESHOLD, NMS_THRESHOLD, debug_frame_count=frame_count)
        elif model_type == 'tflite' and tflite_interpreter:
            input_details_single = tflite_input_details[0]
            is_int8_input = input_details_single['dtype'] == np.int8
            blob_fp32, scale_l, dw_l, dh_l = preprocess_frame_onnx_tflite(frame, MODEL_INPUT_SIZE)
            input_data_to_set = None
            if is_int8_input:
                input_quant_params = input_details_single.get('quantization_parameters', {})
                input_scale_val, input_zero_point_val = input_quant_params.get('scales', [1.0])[0], \
                input_quant_params.get('zero_points', [0])[0]
                if frame_count < 2: print(
                    f"TFLite вхід INT8. Квант. параметри: scale={input_scale_val:.8f}, zero_point={input_zero_point_val}")
                if abs(input_scale_val) < 1e-9:
                    print("ПОМИЛКА: Вхідний масштаб INT8 TFLite близький до нуля!"); input_data_to_set = blob_fp32
                else:
                    input_data_to_set = np.clip((blob_fp32 / input_scale_val) + input_zero_point_val, -128, 127).astype(
                        np.int8)
            else:
                if frame_count < 2: print("TFLite вхід FLOAT32.")
                input_data_to_set = blob_fp32.astype(np.float32)
            tflite_interpreter.set_tensor(input_details_single['index'], input_data_to_set)
            tflite_interpreter.invoke()
            if frame_count < 2: print(f"\n--- TFLite Output Debug Info (Кадр {frame_count}) ---"); print(
                f"К-ть вих. тензорів: {len(tflite_output_details)}")
            output_data_list_fp32 = []
            for i in range(len(tflite_output_details)):
                output_tensor_raw = tflite_interpreter.get_tensor(tflite_output_details[i]['index'])
                dtype, shape, name, quant_params = tflite_output_details[i]['dtype'], tflite_output_details[i]['shape'], \
                tflite_output_details[i]['name'], tflite_output_details[i].get('quantization_parameters', {})
                if frame_count < 2: print(
                    f"  Вих. тензор {i}: name='{name}', shape={shape}, dtype={dtype}, quant_params={quant_params}")
                current_output_fp32 = None
                if dtype == np.int8:
                    output_scale_val, output_zero_point_val = quant_params.get('scales', [1.0])[0], \
                    quant_params.get('zero_points', [0])[0]
                    if frame_count < 2:
                        print(
                            f"    Деквант. INT8 вих. тензор {i} з scale={output_scale_val:.8f}, zero_point={output_zero_point_val}")
                        if output_tensor_raw.size > 0 and output_tensor_raw.shape[1] == 5:  # Якщо це (1,5,8400)
                            print(
                                f"      Сирі INT8 значення для атрибута впевненості (перші 10 з 8400): {output_tensor_raw[0, 4, :10]}")
                    if abs(output_scale_val) < 1e-9:
                        print(
                            f"    ПОМИЛКА: Вих. масштаб INT8 тензора {i} близький до нуля!"); current_output_fp32 = output_tensor_raw.astype(
                            np.float32)
                    else:
                        current_output_fp32 = (output_tensor_raw.astype(
                            np.float32) - output_zero_point_val) * output_scale_val
                else:
                    current_output_fp32 = output_tensor_raw.astype(np.float32)
                output_data_list_fp32.append(current_output_fp32)
            if frame_count < 2: print("--- Кінець TFLite Output Debug Info ---\n")
            primary_output_fp32_for_postprocess = output_data_list_fp32[0] if output_data_list_fp32 and \
                                                                              output_data_list_fp32[
                                                                                  0] is not None else np.array([])
            if frame_count < 2 and primary_output_fp32_for_postprocess.size > 0: print(
                f"Вих. тензор '{tflite_output_details[0]['name']}' форми {primary_output_fp32_for_postprocess.shape} передається в постобробку.")
            current_boxes, current_confidences, current_class_ids = postprocess_yolo_output(
                [primary_output_fp32_for_postprocess], original_shape_hw, MODEL_INPUT_SIZE,
                scale_l, dw_l, dh_l, current_conf_threshold_for_postprocess, NMS_THRESHOLD,
                debug_frame_count=frame_count)

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        if frame_count > 0: total_fps += fps
        frame_count += 1
        frame_with_detections = draw_detections(frame.copy(), current_boxes, current_confidences, current_class_ids,
                                                CLASS_NAMES)
        cv2.putText(frame_with_detections, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(f"Real-time Detection - {model_type.upper()}", frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    avg_fps = total_fps / (frame_count - 1) if frame_count > 1 else total_fps / frame_count if frame_count > 0 else 0
    print(f"Завершено. Середній FPS (без першого кадру, якщо >1): {avg_fps:.2f} для моделі {model_type.upper()}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестування моделей YOLO (.pt, .onnx, .tflite) на відео.")
    parser.add_argument("--video", type=str, required=True, help="Шлях до відеофайлу.")
    parser.add_argument("--model_type", type=str, required=True, choices=['pt', 'onnx', 'tflite'],
                        help="Тип моделі (pt, onnx, tflite).")
    parser.add_argument("--pt_path", type=str, default=None, help="Шлях до .pt моделі (Ultralytics).")
    parser.add_argument("--onnx_path", type=str, default=None, help="Шлях до .onnx моделі.")
    parser.add_argument("--tflite_path", type=str, default=None, help="Шлях до .tflite моделі.")
    args = parser.parse_args()
    if not (ULTRALYTICS_AVAILABLE or ONNXRUNTIME_AVAILABLE or TENSORFLOW_AVAILABLE):
        print("Жодна з необхідних бібліотек для інференсу не встановлена.");
        print("Встановіть: pip install ultralytics onnxruntime tensorflow opencv-python numpy")
    else:
        main(args.video, args.model_type, args.pt_path, args.onnx_path, args.tflite_path)