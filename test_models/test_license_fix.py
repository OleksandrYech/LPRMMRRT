#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_license_fix_litert.py
==========================

Один файл — усе необхідне, щоб протестувати OCR-YOLO моделі
(PyTorch *.pt, ONNX *.onnx, TFLite *.tflite) із використанням
**LiteRT** (ai_edge_litert) для TFLite-інференсу.

• Першою у файлі йде parse_tflite_out() — зручно дебажити.
• Виходи всіх форматів нормалізуються до [0, input_size] і
  проходять 1-D IoU-NMS уздовж осі X, щоб прибрати «стіну нулів».
• Підтримуються пороги `--conf`, `--iou_thr`, кількість запусків `--runs`.

--------------------------------------------------------------------
Приклад:

python3 test_license_fix_litert.py \
        -m detection/models/ocr.pt \
        -m detection/models/ocr_int8.tflite \
        --image test_models/test.png \
        --input_size 320 \
        --conf 0.10 \
        --iou_thr 0.35 \
        --runs 5
--------------------------------------------------------------------
"""
from __future__ import annotations

import argparse, math, os, sys, time
from typing import List, Sequence, Tuple

import cv2, numpy as np
from ultralytics import YOLO
import tensorflow as tf   # тільки для np.float32 підтримки

# ─────────────────────── parse_tflite_out (першим!) ─────────────────────────
def parse_tflite_out(interp, od, thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float):
    """
    Повертає [(x_center_px, char, conf), …] для LiteRT/TFLite-Interpreter.
    Формати: 4-тензорний, 6-атрибутний, сирий YOLO (5+nc).
    Застосовує 1-D IoU-NMS по X.
    """

    def deq(arr, d):
        if arr.dtype == np.float32:
            return arr
        qp = d["quantization_parameters"]
        return (arr.astype(np.float32) - qp["zero_points"]) * qp["scales"]

    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        return inter / ((a1[1]-a1[0]) + (a2[1]-a2[0]) - inter + 1e-6)

    raw_det: List[Tuple[float,float,str,float]] = []

    # 1) 4-тензорний (NMS уже виконано, координати у пікселях)
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        boxes   = deq(interp.get_tensor(od[0]["index"]), od[0])[0]
        scores  = deq(interp.get_tensor(od[1]["index"]), od[1])[0]
        classes = deq(interp.get_tensor(od[2]["index"]), od[2])[0]
        n = int(interp.get_tensor(od[3]["index"])[0])
        for j in range(n):
            cf = float(scores[j]); cid = int(classes[j])
            if cf < thr or cid >= len(names):           continue
            x1, _, x2, _ = boxes[j]
            raw_det.append((x1, x2, names[cid], cf))

    else:
        raw = deq(interp.get_tensor(od[0]["index"]), od[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]

        # 2) «N×6»
        if attrs == 6:
            for x1, _, x2, _, cf, cid in raw[0]:
                cf = float(cf); cid = int(cid)
                if cf < thr or cid >= len(names):        continue
                raw_det.append((x1, x2, names[cid], cf))

        # 3) сирий YOLO «N×(5+nc)»
        elif attrs >= 5 + len(names):
            sigm = lambda x: 1/(1+math.exp(-x))
            for row in raw[0]:
                obj = sigm(float(row[4]))
                if obj < 1e-6:                           continue
                logits = row[5:5+len(names)]
                cid = int(np.argmax(logits))
                cf = obj * sigm(float(logits[cid]))
                if cf < thr or cid >= len(names):        continue
                cx, w = float(row[0])*img_w, float(row[2])*img_w
                if w < 2:                                continue
                x1, x2 = cx - w/2, cx + w/2
                raw_det.append((x1, x2, names[cid], cf))

    if not raw_det:
        return []

    # Масштабування, якщо x2 виходить за межі кадру >20 %
    max_x2 = max(d[1] for d in raw_det)
    if max_x2 > img_w * 1.2:
        scale = img_w / max_x2
        raw_det = [(x1*scale, x2*scale, ch, cf) for x1,x2,ch,cf in raw_det]

    # 1-D IoU-NMS
    raw_det.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float,float,str,float]] = []
    for cand in raw_det:
        if all(iou1d((cand[0],cand[1]), (k[0],k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0]+d[1])/2)
    return [((k[0]+k[1])/2, k[2], k[3]) for k in keep]

# ─────────────────── залишок скрипта (LiteRT + інше) ────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess_tflite(img, inp_det, img_sz):
    shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
    nchw = shp[1] == 3
    h = shp[2] if nchw else shp[1] or img_sz
    w = shp[3] if nchw else shp[2] or img_sz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]
    if dtype in (np.uint8, np.int8):
        qp = inp_det[0]["quantization_parameters"]
        img = (img / (qp["scales"][0] or 1.0) + qp["zero_points"][0]).astype(dtype)
    else:
        img = img.astype(dtype)
    return img

def order_and_string(dets):
    if not dets:
        return "НЕ РОЗПІЗНАНО", 0.0
    dets.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, cf) for _, c, cf in dets])
    return "".join(chars), float(np.mean(confs))

def load_class_names(pt_path):
    if pt_path:
        try:
            names = YOLO(pt_path).names
            if names:
                return names
        except Exception:
            pass
    return CLASS_NAMES

def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr)
    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        r = m(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)
        if i == runs-1 and r.boxes:
            for b in r.boxes:
                cf = float(b.conf.squeeze()); cid = int(b.cls.squeeze())
                if cf < thr or cid >= len(names):          continue
                xc = float(b.xywh.squeeze()[0])
                last.append((xc, names[cid], cf))
    plate, c = order_and_string(last)
    return plate, c, np.mean(times)

def run_litert(path, img, names, thr, runs, img_sz, iou_thr):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ModuleNotFoundError:
        sys.exit("❌ Пакунок ai_edge_litert не встановлено: pip install ai-edge-litert")

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    inp_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess_tflite(img, inp_det, img_sz)
    it.set_tensor(inp_det[0]["index"], inp); it.invoke()   # прогрів

    times, last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(inp_det[0]["index"], inp); it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs-1:
            last = parse_tflite_out(it, out_det, thr, names,
                                    img_w=img_sz, iou_thr=iou_thr)
    plate, c = order_and_string(last)
    return plate, c, np.mean(times)

# ─────────────────────────── CLI та main ────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("OCR-YOLO tester (LiteRT)")
    p.add_argument("-m", "--model", "--path", dest="model_paths",
                   action="append", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--input_size", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--iou_thr", type=float, default=0.35)
    p.add_argument("--runs", type=int, default=5)
    return p.parse_args()

if __name__ == "__main__":
    args = cli()

    for f in args.model_paths + [args.image]:
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: {f}")
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    pt_first = next((p for p in args.model_paths if p.endswith(".pt")), None)
    names = load_class_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model_paths:
        ext = os.path.splitext(mp)[1].lower()
        if ext in (".pt", ".onnx"):
            plate, conf, t = run_pt_or_onnx(mp, img, names,
                                            args.conf, args.runs)
        elif ext == ".tflite":
            plate, conf, t = run_litert(mp, img, names, args.conf, args.runs,
                                        img_sz=args.input_size,
                                        iou_thr=args.iou_thr)
        else:
            print(f"[{os.path.basename(mp):>12}]  ❌ Формат не підтримується.")
            continue
        print(f"[{os.path.basename(mp):>12}]  Plate: {plate:<15} "
              f"Avg conf: {conf:.3f}  Time: {t*1000:.1f} ms")
    print("====================================\n")
