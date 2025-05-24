#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_tester.py
=============

CLI-інструмент для тестування та порівняння OCR-моделей YOLO у форматах
PyTorch (*.pt), ONNX (*.onnx) та TFLite (*.tflite).

Особливості
-----------
* Працює з будь-якою комбінацією моделей: достатньо вказати одну або кілька
  опцій ``--model``.
* Фіксований словник номерних знаків (36 класів) використовується автоматично,
  якщо з *.pt-моделі не вдається прочитати ``model.names``.
* Підтримує два найпоширеніші формати виходу TFLite (1-тензорний та
  4-тензорний із вбудованим NMS).
* Виводить для кожної моделі: розпізнаний номер, середню впевненість символів
  та середній час інференсу.

Виклик
------
python ocr_tester.py \
       --image  test_models/test.png \
       --model  detection/models/ocr.pt \
       --model  detection/models/ocr.onnx \
       --model  detection/models/ocr_int8.tflite \
       --input_size 320 \
       --conf 0.12 \
       --runs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ------------------------------------------------------------------------- #
#                     ПАРАМЕТРИ НОМЕРНОГО СЛОВНИКА                          #
# ------------------------------------------------------------------------- #
NC: int = 36
CLASS_NAMES: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

# ------------------------------------------------------------------------- #
#                            ARGPARSE                                       #
# ------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Порівняння OCR-YOLO моделей (PT / ONNX / TFLite)"
    )
    p.add_argument(
        "--model",
        dest="model_paths",
        required=True,
        action="append",
        help="Шлях до моделі. Параметр можна вказувати багато разів.",
    )
    p.add_argument("--image", required=True, help="Зображення з номерним знаком.")
    p.add_argument(
        "--input_size",
        type=int,
        default=320,
        help="Боковий розмір квадратного входу моделі (H=W).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Поріг confidence для вибору символів.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Кількість повторів для усереднення часу.",
    )
    return p.parse_args()


# ------------------------------------------------------------------------- #
#                          ДОПОМІЖНІ ФУНКЦІЇ                                #
# ------------------------------------------------------------------------- #


def load_class_names(pt_path: str | None) -> List[str]:
    """
    Повертає список імен класів. Спроба прочитати з *.pt, інакше константа.
    """
    if pt_path:
        try:
            model_pt = YOLO(pt_path)
            if getattr(model_pt, "names", None):
                return model_pt.names
        except Exception:
            pass
    return CLASS_NAMES


def preprocess_tflite(
    img_bgr: np.ndarray,
    input_details: Sequence[dict],
    img_sz: int,
) -> np.ndarray:
    """
    Готує зображення під TFLite (NHWC або NCHW, float32 / int8 / uint8).
    """
    in_shape = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"]

    # Формально в in_shape можуть бути -1, тому беремо img_sz як fallback.
    if len(in_shape) != 4:
        raise ValueError(f"Непідтримувана форма входу TFLite: {in_shape}")

    is_nchw = in_shape[1] == 3  # (1,3,H,W)
    target_h = in_shape[2] if is_nchw else in_shape[1]
    target_w = in_shape[3] if is_nchw else in_shape[2]
    if target_h <= 0 or target_w <= 0:
        target_h = target_w = img_sz

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img_f32 = img_resized.astype(np.float32) / 255.0
    if is_nchw:
        img_f32 = np.transpose(img_f32, (2, 0, 1))  # HWC → CHW
    img_f32 = np.expand_dims(img_f32, axis=0)  # (1, …)

    # Квантуємо, якщо потрібно
    if in_dtype in (np.int8, np.uint8):
        quant = input_details[0]["quantization_parameters"]
        scale = quant["scales"][0] if quant["scales"].size else 1.0
        zero_pt = quant["zero_points"][0] if quant["zero_points"].size else 0
        img_q = (img_f32 / scale + zero_pt).astype(in_dtype)
        return img_q

    return img_f32.astype(in_dtype)


def order_and_stringify(
    detections: List[Tuple[float, str, float]]
) -> Tuple[str, float]:
    """
    Сортує символи за координатою X і повертає
    (рядок_номера, середня_впевненість).
    """
    if not detections:
        return "НЕ РОЗПІЗНАНО", 0.0
    detections.sort(key=lambda d: d[0])  # за x-центром
    chars = [d[1] for d in detections]
    confs = [d[2] for d in detections]
    return "".join(chars), float(np.mean(confs))


# ------------------------------------------------------------------------- #
#                              BACK-ENDS                                    #
# ------------------------------------------------------------------------- #


def run_pt_or_onnx(
    model_path: str,
    img_bgr: np.ndarray,
    class_names: Sequence[str],
    conf_thr: float,
    runs: int,
) -> Tuple[str, float, float]:
    """
    Інференс *.pt або *.onnx через Ultralytics.
    """
    model = YOLO(model_path)
    times: List[float] = []
    last_det: List[Tuple[float, str, float]] = []

    _ = model(img_bgr, verbose=False, conf=conf_thr)  # прогрів

    for i in range(runs):
        t0 = time.perf_counter()
        res = model(img_bgr, verbose=False, conf=conf_thr)[0]
        times.append(time.perf_counter() - t0)

        if i == runs - 1 and res.boxes:
            for box in res.boxes:
                cls_id = int(box.cls.squeeze())
                conf = float(box.conf.squeeze())
                if conf < conf_thr or cls_id >= len(class_names):
                    continue
                char = class_names[cls_id]
                x_center = float(box.xywh.squeeze()[0])
                last_det.append((x_center, char, conf))

    plate, avg_conf = order_and_stringify(last_det)
    return plate, avg_conf, float(np.mean(times))


def _parse_tflite_outputs(
    interpreter: tf.lite.Interpreter,
    output_details: Sequence[dict],
    conf_thr: float,
    class_names: Sequence[str],
) -> List[Tuple[float, str, float]]:
    """
    Повертає список детекцій (x_center, char, conf)
    для двох підтримуваних форматів виходу.
    """
    detections: List[Tuple[float, str, float]] = []

    # Формат 4-тензорний: boxes, scores, classes, count
    if (
        len(output_details) == 4
        and output_details[0]["shape"].ndim == 3
        and output_details[0]["shape"][-1] == 4
    ):
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]      # (N,4)
        scores = interpreter.get_tensor(output_details[1]["index"])[0]     # (N,)
        classes = interpreter.get_tensor(output_details[2]["index"])[0]    # (N,)
        n_det = int(interpreter.get_tensor(output_details[3]["index"])[0]) # scalar

        for j in range(n_det):
            conf = float(scores[j])
            if conf < conf_thr:
                continue
            cls_id = int(classes[j])
            if cls_id >= len(class_names):
                continue
            char = class_names[cls_id]
            x1, y1, x2, y2 = boxes[j]  # вже у пікселях (Ultralytics експорти)
            xc = (x1 + x2) / 2.0
            detections.append((xc, char, conf))
        return detections

    # Формат 1-тензорний: [1, N, ≥6]
    raw = interpreter.get_tensor(output_details[0]["index"])
    if raw.ndim == 3 and raw.shape[2] >= 6:
        for det in raw[0]:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            conf = float(conf)
            if conf < conf_thr:
                continue
            cls_id = int(cls_id)
            if cls_id >= len(class_names):
                continue
            char = class_names[cls_id]
            xc = (x1 + x2) / 2.0
            detections.append((xc, char, conf))
    return detections


def run_tflite(
    model_path: str,
    img_bgr: np.ndarray,
    class_names: Sequence[str],
    conf_thr: float,
    runs: int,
    img_sz: int,
) -> Tuple[str, float, float]:
    """
    Інференс *.tflite (INT8 або float32).
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    inp = preprocess_tflite(img_bgr, inp_det, img_sz)

    times: List[float] = []
    last_det: List[Tuple[float, str, float]] = []

    # прогрів
    interpreter.set_tensor(inp_det[0]["index"], inp)
    interpreter.invoke()

    for i in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp_det[0]["index"], inp)
        interpreter.invoke()
        times.append(time.perf_counter() - t0)

        if i == runs - 1:
            last_det = _parse_tflite_outputs(
                interpreter, out_det, conf_thr, class_names
            )

    plate, avg_conf = order_and_stringify(last_det)
    return plate, avg_conf, float(np.mean(times))


# ------------------------------------------------------------------------- #
#                                 MAIN                                      #
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = parse_args()

    # --- Перевірка існування файлів ---
    for path in args.model_paths + [args.image]:
        if not os.path.exists(path):
            sys.exit(f"Файл не знайдено: {path}")

    # --- Зображення ---
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        sys.exit(f"Не вдалося відкрити зображення {args.image}")

    # --- Імена класів ---
    pt_models = [p for p in args.model_paths if p.lower().endswith(".pt")]
    class_names = load_class_names(pt_models[0]) if pt_models else CLASS_NAMES

    # --- Обхід моделей ---
    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for model_path in args.model_paths:
        ext = os.path.splitext(model_path)[1].lower()

        if ext in (".pt", ".onnx"):
            plate, conf, t = run_pt_or_onnx(
                model_path, image_bgr, class_names, args.conf, args.runs
            )

        elif ext == ".tflite":
            plate, conf, t = run_tflite(
                model_path,
                image_bgr,
                class_names,
                args.conf,
                args.runs,
                img_sz=args.input_size,
            )

        else:
            print(f"[{os.path.basename(model_path):>12}]  ❌ Невідомий формат, пропущено.")
            continue

        print(
            f"[{os.path.basename(model_path):>12}]  "
            f"Plate: {plate:<15}  "
            f"Avg conf: {conf:.3f}  "
            f"Time: {t*1000:.1f} ms"
        )
    print("====================================\n")
