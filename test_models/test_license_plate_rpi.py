#!/usr/bin/env python3
"""
ocr_tester.py

Зручний CLI-скрипт для порівняння трьох форматів моделей OCR-YOLO
(Pytorch *.pt, ONNX *.onnx, TFLite *.tflite INT8) на одному зображенні.

Запуск:
    python ocr_tester.py --image /path/to/img.png \
                         --model /path/to/ocr.pt \
                         --model /path/to/ocr.onnx \
                         --model /path/to/ocr.tflite \
                         --input_size 320 \
                         --conf 0.12 \
                         --runs 5
"""

import argparse
import sys
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ------------------------------- CLI -----------------------------------------


def parse_args() -> argparse.Namespace:
    """Парсинг аргументів командного рядка."""
    p = argparse.ArgumentParser(
        description="Тест OCR-моделей YOLO у форматах PyTorch / ONNX / TFLite"
    )
    p.add_argument(
        "--model",
        required=True,
        action="append",
        dest="model_paths",
        help="Шлях до моделі. Параметр повторюється для кожної моделі.",
    )
    p.add_argument("--image", required=True, help="Шлях до зображення з номером.")
    p.add_argument(
        "--input_size",
        type=int,
        default=320,
        help="Розмір (H=W) на вході моделі. За замовч. 320.",
    )
    p.add_argument(
        "--conf", type=float, default=0.12, help="Поріг confidence для символів."
    )
    p.add_argument(
        "--runs", type=int, default=5, help="Кількість повторів для усереднення часу."
    )
    return p.parse_args()


# ------------------------------ Utils ----------------------------------------


def load_class_names(pt_path: str) -> List[str]:
    """
    Читає імена класів із *.pt-моделі Ultralytics.
    Потрібно для TFLite/ONNX, де names не зберігаються.
    """
    try:
        model_pt = YOLO(pt_path)
        if hasattr(model_pt, "names") and model_pt.names:
            return model_pt.names
    except Exception:
        pass
    sys.exit(
        f"Не вдалося отримати імена класів із {pt_path}. "
        "Запустіть зі своїм .pt-файлом або задайте class_names вручну."
    )


def preprocess_tflite(
    img_bgr: np.ndarray, input_details: List[dict], img_sz: int
) -> np.ndarray:
    """
    Приводить зображення до формату TFLite-моделі, підтримуючи NHWC та NCHW,
    float32 / int8 / uint8.
    """
    # Очікуваний Shape
    in_shape = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"]

    # Визначаємо порядок каналів
    is_nchw = len(in_shape) == 4 and in_shape[1] == 3
    target_h = in_shape[2] if is_nchw else in_shape[1]
    target_w = in_shape[3] if is_nchw else in_shape[2]

    # Resize + RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h))

    # Нормалізація 0‒1
    img_f32 = img_resized.astype(np.float32) / 255.0
    if is_nchw:
        img_f32 = np.transpose(img_f32, (2, 0, 1))  # HWC→CHW

    img_f32 = np.expand_dims(img_f32, 0)  # (1, …)

    # Квантування при потребі
    if in_dtype in (np.uint8, np.int8):
        scale = input_details[0]["quantization_parameters"]["scales"][0]
        zero_pt = input_details[0]["quantization_parameters"]["zero_points"][0]
        img_q = (img_f32 / scale + zero_pt).astype(in_dtype)
        return img_q
    return img_f32.astype(in_dtype)


def order_and_stringify(
    detections: List[Tuple[float, str, float]]
) -> Tuple[str, float]:
    """
    Сортує символи за координатою X та формує строку з номером.
    Повертає (рядок, середня_впевненість).
    """
    if not detections:
        return "НЕ РОЗПІЗНАНО", 0.0
    detections.sort(key=lambda d: d[0])  # за x
    chars = [d[1] for d in detections]
    confs = [d[2] for d in detections]
    return "".join(chars), float(np.mean(confs))


# ---------------------------- Inference back-ends -----------------------------


def run_pt_or_onnx(
    model_path: str,
    img_bgr: np.ndarray,
    class_names: List[str],
    conf_thr: float,
    runs: int,
) -> Tuple[str, float, float]:
    """Запуск *.pt або *.onnx через Ultralytics."""
    model = YOLO(model_path)
    times, last_detections = [], []

    # Заздалегідь один прогрів
    _ = model(img_bgr, verbose=False, conf=conf_thr)

    for i in range(runs):
        t0 = time.perf_counter()
        res = model(img_bgr, verbose=False, conf=conf_thr)[0]
        times.append(time.perf_counter() - t0)

        if i == runs - 1 and res.boxes:
            for box in res.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                char = (
                    class_names[cls_id] if cls_id < len(class_names) else "UNK"
                )
                xc = float(box.xywh[0])
                last_detections.append((xc, char, conf))

    plate_str, avg_conf = order_and_stringify(last_detections)
    return plate_str, avg_conf, float(np.mean(times))


def run_tflite(
    model_path: str,
    img_bgr: np.ndarray,
    class_names: List[str],
    conf_thr: float,
    runs: int,
) -> Tuple[str, float, float]:
    """Запуск *.tflite INT8 з Ultralytics-експортом (NMS всередині)."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

    inp = preprocess_tflite(img_bgr, input_details, img_sz=args.input_size)

    times, last_detections = [], []
    # Прогрів
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    for i in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details[0]["index"])  # [1,N,6]
        times.append(time.perf_counter() - t0)

        if i == runs - 1 and raw.ndim == 3 and raw.shape[2] >= 6:
            for x1, y1, x2, y2, conf, cls in raw[0]:
                if conf < conf_thr:
                    continue
                cls_id = int(cls)
                char = class_names[cls_id] if cls_id < len(class_names) else "UNK"
                x_center = (x1 + x2) / 2.0  # Уже у пікселях (Ultralytics export)
                last_detections.append((x_center, char, float(conf)))

    plate_str, avg_conf = order_and_stringify(last_detections)
    return plate_str, avg_conf, float(np.mean(times))


# -------------------------------- Main ---------------------------------------


if __name__ == "__main__":
    args = parse_args()
    if not args.model_paths:
        sys.exit("Не вказано жодної моделі (--model).")

    # Перевірка файлів
    for p in args.model_paths + [args.image]:
        if not os.path.exists(p):
            sys.exit(f"Файл не знайдено: {p}")

    # Головне зображення
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити зображення {args.image}")

    # Імена класів (обов’язково) — беремо з першої *.pt
    pt_files = [p for p in args.model_paths if p.lower().endswith(".pt")]
    if not pt_files:
        sys.exit("Потрібен хоча б один .pt для завантаження імен класів.")
    names = load_class_names(pt_files[0])

    #   ­— Запуск кожної моделі —
    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model_paths:
        ext = os.path.splitext(mp)[1].lower()
        if ext in (".pt", ".onnx"):
            plate, conf, tm = run_pt_or_onnx(
                mp, img, names, args.conf, args.runs
            )
        elif ext == ".tflite":
            plate, conf, tm = run_tflite(
                mp, img, names, args.conf, args.runs
            )
        else:
            print(f"[{mp}] ‒ Пропущено: невідомий формат")
            continue

        print(
            f"[{os.path.basename(mp):>12}]  "
            f"Plate: {plate:<15}  "
            f"Avg conf: {conf:.3f}  "
            f"Time: {tm*1000:.1f} ms"
        )
    print("====================================\n")
