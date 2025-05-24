#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_license_fix.py
===================

CLI-тестер OCR-YOLO моделей у форматах:
* PyTorch *.pt
* ONNX *.onnx
* TFLite *.tflite (INT8 / FP32, з різними схемами експорту)

Виводить: розпізнаний номер, середню впевненість символів, середній час.

Запуск (приклад):
-----------------
python3 test_license_fix.py \
        --model detection/models/ocr.pt \
        --model detection/models/ocr_int8.tflite \
        --image test_models/test.png \
        --input_size 320 \
        --conf 0.10 \
        --runs 5
"""

from __future__ import annotations
import argparse, math, os, sys, time
from typing import List, Sequence, Tuple

import cv2, numpy as np
from ultralytics import YOLO
import tensorflow as tf

# --------------------------- словник номерів --------------------------------
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # nc = 36

# --------------------------- аргументи CLI ----------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Тестування OCR-YOLO моделей")
    p.add_argument(
        "-m", "--model", "--path", dest="model_paths",
        action="append", required=True,
        help="Шлях до моделі; опцію можна вказувати багато разів."
    )
    p.add_argument("--image", required=True, help="Зображення з номером.")
    p.add_argument("--input_size", type=int, default=320, help="H=W входу.")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг confidence.")
    p.add_argument("--runs", type=int, default=5, help="Повторів для часу.")
    return p.parse_args()

# --------------------------- утиліти ----------------------------------------
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

def order_and_stringify(det: List[Tuple[float, str, float]]):
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])                  # x-центр ↑
    chars, confs = zip(*[(c, f) for _, c, f in det])
    return "".join(chars), float(np.mean(confs))

# --------------------------- парсер виходу TFLite ---------------------------
def parse_tflite_out(interp, od, thr: float, names,
                     img_w: int = 320, iou_thr: float = 0.3):
    """Повертає [(x_center_px, char, conf), …] з 1-D IoU-NMS."""

    def dequant(arr: np.ndarray, od_: dict) -> np.ndarray:
        if arr.dtype == np.float32:
            return arr
        qp = od_["quantization_parameters"]
        return (arr.astype(np.float32) - qp["zero_points"]) * qp["scales"]

    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        union = (a1[1] - a1[0]) + (a2[1] - a2[0]) - inter
        return inter / union if union else 0.0

    raw_det: List[Tuple[float, float, str, float]] = []

    # ---------- 4-тензорний вихід ------------------------------------------
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        boxes   = dequant(interp.get_tensor(od[0]["index"]), od[0])[0]
        scores  = dequant(interp.get_tensor(od[1]["index"]), od[1])[0]
        classes = dequant(interp.get_tensor(od[2]["index"]), od[2])[0]
        n = int(interp.get_tensor(od[3]["index"])[0])

        for j in range(n):
            conf = float(scores[j])
            if conf < thr:                     continue
            cid = int(classes[j])
            if cid >= len(names):              continue
            x1, _, x2, _ = boxes[j]
            raw_det.append((x1, x2, names[cid], conf))

    # ---------- 1-тензорні формати -----------------------------------------
    else:
        raw = dequant(interp.get_tensor(od[0]["index"]), od[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]

        # 6-атрибутний
        if attrs == 6:
            for x1, _, x2, _, conf, cid in raw[0]:
                conf = float(conf); cid = int(cid)
                if conf < thr or cid >= len(names): continue
                raw_det.append((x1, x2, names[cid], conf))

        # сирий YOLO (5+nc)
        elif attrs >= 5 + len(names):
            sigm = lambda x: 1.0 / (1.0 + math.exp(-x))
            for row in raw[0]:
                obj = sigm(float(row[4]))
                if obj < 1e-6:                continue
                logits = row[5 : 5 + len(names)]
                cid = int(np.argmax(logits))
                conf = obj * sigm(float(logits[cid]))
                if conf < thr or cid >= len(names): continue
                cx = float(row[0]) * img_w
                w  = float(row[2]) * img_w
                x1, x2 = cx - w / 2.0, cx + w / 2.0
                raw_det.append((x1, x2, names[cid], conf))

    if not raw_det:
        return []

    # ---------- 1-D IoU-NMS -----------------------------------------------
    raw_det.sort(key=lambda d: d[3], reverse=True)      # conf ↓
    keep: List[Tuple[float, float, str, float]] = []
    for cand in raw_det:
        if all(iou1d((cand[0], cand[1]), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0] + d[1]) / 2.0)        # x-центр ↑
    return [((k[0] + k[1]) / 2.0, k[2], k[3]) for k in keep]

# --------------------------- back-ends --------------------------------------
def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr)
    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        r = m(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)
        if i == runs - 1 and r.boxes:
            for b in r.boxes:
                cf = float(b.conf.squeeze())
                if cf < thr: continue
                cid = int(b.cls.squeeze())
                if cid >= len(names): continue
                xc = float(b.xywh.squeeze()[0])
                last.append((xc, names[cid], cf))
    plate, c = order_and_stringify(last)
    return plate, c, np.mean(times)

def run_tflite(path, img, names, thr, runs, img_sz):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess_tflite(img, inp_det, img_sz)
    it.set_tensor(inp_det[0]["index"], inp); it.invoke()      # прогрів
    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(inp_det[0]["index"], inp); it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs - 1:
            last = parse_tflite_out(it, out_det, thr, names, img_w=img_sz)
    plate, c = order_and_stringify(last)
    return plate, c, np.mean(times)

# --------------------------- main ------------------------------------------
if __name__ == "__main__":
    args = parse_args()

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
