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

import math
import numpy as np

def parse_tflite_out(interp, out_det, thr: float, names, img_w: int = 320,
                     x_tol_px: float = 6.0):
    """
    Повертає [(x_center_px, char, conf), …] із коректним масштабуванням
    та 1-D NMS по осі X.  `img_w` — ширина входу моделі (звично 320).
    """

    def dequant(arr: np.ndarray, od: dict) -> np.ndarray:
        if arr.dtype == np.float32:
            return arr
        qp = od["quantization_parameters"]
        return (arr.astype(np.float32) - qp["zero_points"]) * qp["scales"]

    det_raw = []                      # сирі детекції
    keep = []                         # після NMS-1D

    # ---------- 1) 4-тензорний вихід --------------------------------------
    if len(out_det) == 4 and out_det[0]["shape"][-1] == 4:
        boxes   = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])[0]
        scores  = dequant(interp.get_tensor(out_det[1]["index"]), out_det[1])[0]
        classes = dequant(interp.get_tensor(out_det[2]["index"]), out_det[2])[0]
        n = int(interp.get_tensor(out_det[3]["index"])[0])

        for j in range(n):
            conf = float(scores[j])
            if conf < thr:
                continue
            cls = int(classes[j])
            if cls >= len(names):
                continue
            x1, _, x2, _ = boxes[j]
            xc_px = (x1 + x2) / 2.0              # уже у пікселях
            det_raw.append((xc_px, names[cls], conf))

    # ---------- 2) один великий тензор ------------------------------------
    else:
        raw = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if raw.ndim != 3:
            return []

        attrs = raw.shape[2]

        # ― 2.a «N×6» ------------------------------------------------------
        if attrs == 6:
            for x1, _, x2, _, conf, cls in raw[0]:
                conf = float(conf); cls = int(cls)
                if conf < thr or cls >= len(names):
                    continue
                xc_px = (x1 + x2) / 2.0          # у пікселях
                det_raw.append((xc_px, names[cls], conf))

        # ― 2.b «N×(5+nc)» сирий YOLO -------------------------------------
        elif attrs >= 5 + len(names):
            sigm = lambda x: 1.0 / (1.0 + math.exp(-x))
            for row in raw[0]:
                obj = sigm(float(row[4]))
                if obj < 1e-6:
                    continue
                cls_logits = row[5 : 5 + len(names)]
                cls = int(np.argmax(cls_logits))
                cls_prob = sigm(float(cls_logits[cls]))
                conf = obj * cls_prob
                if conf < thr or cls >= len(names):
                    continue
                cx_norm = float(row[0])          # 0-1
                xc_px = cx_norm * img_w          # → пікселі!
                det_raw.append((xc_px, names[cls], conf))

    # ---------- 1-D NMS уздовж X-центру -----------------------------------
    det_raw.sort(key=lambda d: d[2], reverse=True)        # по conf
    for xc, ch, cf in det_raw:
        if all(abs(xc - k[0]) > x_tol_px for k in keep):
            keep.append((xc, ch, cf))

    return keep

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
