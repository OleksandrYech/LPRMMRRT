#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_tester.py — CLI-тестер OCR-YOLO моделей (PT / ONNX / TFLite).
"""

from __future__ import annotations
import argparse, os, sys, time
from typing import List, Sequence, Tuple
import cv2, numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ------------------ словник номерів (36 класів) ----------------------------
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # len == 36

# ------------------ аргументи командного рядка -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Тестування OCR-YOLO моделей")
    p.add_argument("--model", action="append", required=True,
                   help="Шлях до моделі; опцію можна повторювати.")
    p.add_argument("--image", required=True,
                   help="Зображення з номерним знаком.")
    p.add_argument("--input_size", type=int, default=320,
                   help="Квадратний розмір входу (H=W).")
    p.add_argument("--conf", type=float, default=0.12,
                   help="Поріг confidence символів.")
    p.add_argument("--runs", type=int, default=5,
                   help="Скільки разів запускати для усереднення часу.")
    return p.parse_args()

# ------------------ утиліти -------------------------------------------------
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
    shape = inp_det[0]["shape"]; dtype = inp_det[0]["dtype"]
    nchw = shape[1] == 3
    h = shape[2] if nchw else shape[1] or img_sz
    w = shape[3] if nchw else shape[2] or img_sz

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]  # (1,…)

    if dtype in (np.uint8, np.int8):
        qp = inp_det[0]["quantization_parameters"]
        scale = qp["scales"][0] or 1.0
        zp = qp["zero_points"][0]
        img = (img / scale + zp).astype(dtype)
    else:
        img = img.astype(dtype)
    return img

# ------------------ допоміжні функції --------------------------------------
def order_and_stringify(det):  # det: List[(x, char, conf)]
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, f) for _, c, f in det])
    return "".join(chars), float(np.mean(confs))

def dequant(arr, od):
    """Повертає float32 масив незалежно від dtype."""
    if arr.dtype == np.float32:
        return arr.astype(np.float32)
    qp = od["quantization_parameters"]
    scale = qp["scales"]
    zp = qp["zero_points"]
    if scale.size == 0:                      # деякі edge-TPU моделі
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zp) * scale

# ------------------ інференс back-ends --------------------------------------
def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr)  # прогрів
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

def parse_tflite_out(interp, od, thr, names):
    det = []
    # 4-тензорний формат
    if len(od) == 4 and od[0]["shape"].ndim == 3 and od[0]["shape"][-1] == 4:
        boxes   = dequant(interp.get_tensor(od[0]["index"]), od[0])[0]
        scores  = dequant(interp.get_tensor(od[1]["index"]), od[1])[0]
        classes = dequant(interp.get_tensor(od[2]["index"]), od[2])[0]
        n = int(interp.get_tensor(od[3]["index"])[0])
        for j in range(n):
            conf = float(scores[j]);   cls = int(classes[j])
            if conf < thr or cls >= len(names): continue
            x1, y1, x2, y2 = boxes[j]; xc = (x1 + x2) / 2.0
            det.append((xc, names[cls], conf))
        return det
    # 1-тензорний формат
    raw = dequant(interp.get_tensor(od[0]["index"]), od[0])
    if raw.ndim == 3 and raw.shape[2] >= 6:
        for x1, y1, x2, y2, conf, cls in raw[0]:
            conf = float(conf); cls = int(cls)
            if conf < thr or cls >= len(names): continue
            xc = (x1 + x2) / 2.0
            det.append((xc, names[cls], conf))
    return det

def run_tflite(path, img, names, thr, runs, img_sz):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess_tflite(img, inp_det, img_sz)

    times, last = [], []
    it.set_tensor(inp_det[0]["index"], inp); it.invoke()  # прогрів
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(inp_det[0]["index"], inp); it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs - 1:
            last = parse_tflite_out(it, out_det, thr, names)
    plate, c = order_and_stringify(last)
    return plate, c, np.mean(times)

# ------------------ головна логіка ------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # перевірки
    for p in args.model_paths + [args.image]:
        if not os.path.exists(p): sys.exit(f"Файл не знайдено: {p}")

    img = cv2.imread(args.image)
    if img is None: sys.exit(f"Не вдалося відкрити {args.image}")

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
