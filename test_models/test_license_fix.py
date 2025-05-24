#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_license_fix.py
===================

CLI-тестер OCR-YOLO-моделей (PyTorch, ONNX, TFLite).
"""

from __future__ import annotations
import argparse, os, sys, time
from typing import List, Sequence, Tuple
import cv2, numpy as np
from ultralytics import YOLO
import tensorflow as tf
import math 


# ---------- словник номерних знаків (36 класів) ----------------------------
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------- аргументи командного рядка -------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Тестування OCR-YOLO моделей")
    p.add_argument(
        "-m", "--model", "--path", dest="model_paths",
        action="append", required=True,
        help="Шлях до моделі; опцію можна повторювати."
    )
    p.add_argument("--image", required=True, help="Зображення з номерним знаком.")
    p.add_argument("--input_size", type=int, default=320, help="Розмір входу (H=W).")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг confidence.")
    p.add_argument("--runs", type=int, default=5, help="Кількість повторів.")
    return p.parse_args()


# ---------- допоміжні функції ----------------------------------------------
def load_class_names(pt_path: str | None) -> List[str]:
    if pt_path:
        try:
            names = YOLO(pt_path).names
            if names:
                return names
        except Exception:
            pass
    return CLASS_NAMES


def preprocess_tflite(img, inp_det, img_sz):
    shape, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
    nchw = shape[1] == 3
    h = shape[2] if nchw else shape[1] or img_sz
    w = shape[3] if nchw else shape[2] or img_sz

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]

    if dtype in (np.uint8, np.int8):
        qp = inp_det[0]["quantization_parameters"]
        scale = qp["scales"][0] or 1.0
        zp = qp["zero_points"][0]
        img = (img / scale + zp).astype(dtype)
    else:
        img = img.astype(dtype)
    return img


def dequant(arr, od):
    if arr.dtype == np.float32:
        return arr
    qp = od["quantization_parameters"]
    scale = qp["scales"]
    zp = qp["zero_points"]
    return (arr.astype(np.float32) - zp) * scale


def order_and_stringify(det):                     # det: List[(x, char, conf)]
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, f) for _, c, f in det])
    return "".join(chars), float(np.mean(confs))


# ---------- back-ends --------------------------------------------------------
def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr)
    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        r = m(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)
        if i == runs - 1 and r.boxes:
            for b in r.boxes:
                conf = float(b.conf.squeeze())
                if conf < thr: continue
                cls = int(b.cls.squeeze())
                if cls >= len(names): continue
                xc = float(b.xywh.squeeze()[0])
                last.append((xc, names[cls], conf))
    plate, c = order_and_stringify(last)
    return plate, c, np.mean(times)


import math
import numpy as np

def parse_tflite_out(interp, out_det, thr: float, names):
    """
    Парсить вихідні тензори TFLite-моделі YOLO та повертає
    список детекцій у форматі:
        [(x_center, char, confidence), ...].

    Підтримує три найпоширеніші формати експорту:

    1) **4-тензорний** із вбудованим NMS
       └─ boxes  : (1,N,4)  — [x1,y1,x2,y2]  (у пікселях)
       └─ scores : (1,N)    — conf
       └─ classes: (1,N)    — class_id
       └─ count  : (1,)     — фактична кількість детекцій N

    2) **1-тензорний «N×6»**
       └─ [x1, y1, x2, y2, conf, class_id]
       (координати можуть бути нормалізовані; для сортування
       потрібен лише x-центр, тому масштабування опускаємо).

    3) **1-тензорний «N×(5+nc)»** (сирий YOLO-вихід)
       └─ [cx, cy, w, h, obj_logit, cls_logit0 … cls_logit{nc-1}]
       (логіти → ймовірності через sigmoid; conf = obj * cls_prob).

    Параметри
    ---------
    interp : tf.lite.Interpreter
        Активний інтерпретатор із виконаним `invoke()`.
    out_det : list[dict]
        Список словників із `shape`, `index`, `quantization_parameters`.
    thr : float
        Поріг confidence (у діапазоні 0‒1) для фільтрації символів.
    names : Sequence[str]
        Список імен класів (довжина == nc).

    Повертає
    --------
    list[tuple[float, str, float]]
        Відсортовані за x_center детекції символів.
    """
    def dequant(arr: np.ndarray, od: dict) -> np.ndarray:
        """Переводить INT8/UINT8 → float32; float32 лишає без змін."""
        if arr.dtype == np.float32:
            return arr.astype(np.float32)
        qp = od["quantization_parameters"]
        scales = qp["scales"]
        zero_pts = qp["zero_points"]
        if scales.size == 0:                         # EdgeTPU edge-case
            return arr.astype(np.float32)
        return (arr.astype(np.float32) - zero_pts) * scales

    detections = []

    # ---------- 1. 4-тензорний вихід ---------------------------------------
    if (
        len(out_det) == 4
        and out_det[0]["shape"].ndim == 3
        and out_det[0]["shape"][-1] == 4
    ):
        boxes   = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])[0]
        scores  = dequant(interp.get_tensor(out_det[1]["index"]), out_det[1])[0]
        classes = dequant(interp.get_tensor(out_det[2]["index"]), out_det[2])[0]
        n_det   = int(interp.get_tensor(out_det[3]["index"])[0])

        for j in range(n_det):
            conf = float(scores[j])
            if conf < thr:
                continue
            cls_id = int(classes[j])
            if cls_id >= len(names):
                continue
            x1, _, x2, _ = boxes[j]
            x_center = (x1 + x2) / 2.0
            detections.append((x_center, names[cls_id], conf))
        return detections

    # ---------- 2. Один великий тензор -------------------------------------
    raw = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
    if raw.ndim != 3:
        return detections                       # невідомий формат

    attrs = raw.shape[2]

    # ---- 2.a «N×6» --------------------------------------------------------
    if attrs == 6:
        for x1, _, x2, _, conf, cls_id in raw[0]:
            conf = float(conf)
            cls_id = int(cls_id)
            if conf < thr or cls_id >= len(names):
                continue
            x_center = (x1 + x2) / 2.0
            detections.append((x_center, names[cls_id], conf))
        return detections

    # ---- 2.b «N×(5+nc)» сирий YOLO ---------------------------------------
    #  [cx, cy, w, h, obj_logit, cls_logit0 …]
    if attrs >= 5 + len(names):
        sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
        for row in raw[0]:
            obj_conf = sigmoid(float(row[4]))          # obj_logit → prob
            if obj_conf < 1e-6:                        # швидкий пропуск
                continue
            cls_logits = row[5 : 5 + len(names)]
            cls_id = int(np.argmax(cls_logits))
            cls_prob = sigmoid(float(cls_logits[cls_id]))
            conf = obj_conf * cls_prob
            if conf < thr:
                continue
            x_center = float(row[0])                   # cx уже нормалізовано
            detections.append((x_center, names[cls_id], conf))
    # ----------------------------------------------------------------------
    return detections


def run_tflite(path, img, names, thr, runs, img_sz):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess_tflite(img, inp_det, img_sz)
    it.set_tensor(inp_det[0]["index"], inp); it.invoke()     # прогрів
    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(inp_det[0]["index"], inp); it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs - 1:
            last = parse_tflite_out(it, out_det, thr, names)
    plate, c = order_and_stringify(last)
    return plate, c, np.mean(times)


# ---------- main -------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # перевірка файлів
    for p in args.model_paths + [args.image]:
        if not os.path.exists(p):
            sys.exit(f"Файл не знайдено: {p}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    pt_first = next((p for p in args.model_paths if p.endswith(".pt")), None)
    names = load_class_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for p in args.model_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".pt", ".onnx"):
            plate, conf, t = run_pt_or_onnx(p, img, names, args.conf, args.runs)
        elif ext == ".tflite":
            plate, conf, t = run_tflite(p, img, names, args.conf,
                                        args.runs, args.input_size)
        else:
            print(f"[{os.path.basename(p):>12}]  ❌ Формат не підтримується.")
            continue
        print(f"[{os.path.basename(p):>12}]  Plate: {plate:<15}  "
              f"Avg conf: {conf:.3f}  Time: {t*1000:.1f} ms")
    print("====================================\n")
