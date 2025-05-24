# -*- coding: utf-8 -*-
"""
video_test_fixed.py — універсальний скрипт для тестування YOLO‑моделей у форматах
.pt (Ultralytics), .onnx (ONNX Runtime) та .tflite (TensorFlow Lite).

Зміни проти оригіналу
=====================
1. **Авто‑розмір ONNX** — `MODEL_INPUT_SIZE` береться прямо з `onnx_session.get_inputs()[0]`.
2. **Усунення подвійного множення** у `postprocess_yolo_output`.
   • Якщо координати в ONNX уже у пікселях — не множимо на `net_w`/`net_h`.
3. **Середня впевненість**
   • Коли запущено з `--no_display`, у консоль кожні 100 кадрів виводиться
     середня впевненість (mean confidence) по всіх прийнятих детекціях.
4. **Оптимізація Pre‑process** — обрізано непотрібні привʼязки до глобальних
   констант; функції отримують усе необхідне через параметри.

Запуск прикладу
---------------
python3 video_test_fixed.py \
    --video test.mp4 \
    --model_type onnx \
    --onnx_path detection/models/license.onnx \
    --frame_skip 5 \
    --no_display

"""

import cv2
import numpy as np
import time, argparse, os, sys

# ────────────────── залежності під різні формати ────────────────────
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ────────────────────────── константи ───────────────────────────────
CONF_THRESHOLD = 0.4
NMS_THRESHOLD  = 0.5
CLASS_NAMES    = ['License_Plate']
TARGET_CLASS_ID = 0

# ────────────────────────── utils ───────────────────────────────────

def preprocess_letterbox(frame, dst_size):
    """Letterbox‑resize + повертає blob NCHW 0‑1."""
    net_w, net_h = dst_size
    img_h, img_w = frame.shape[:2]
    scale = min(net_w / img_w, net_h / img_h)
    new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))
    dw, dh = (net_w - new_w) / 2, (net_h - new_h) / 2

    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                                  int(round(dw - 0.1)), int(round(dw + 0.1)),
                                  cv2.BORDER_CONSTANT, value=(114,114,114))
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (net_w, net_h), swapRB=True)
    return blob, scale, dw, dh

# ───────────────────────── postprocess ──────────────────────────────

def postprocess_yolo_output(predictions, orig_hw, net_size, scale, dw, dh):
    """Повертає (boxes xywh, confidences, class_ids)."""
    ih, iw   = orig_hw
    nw, nh   = net_size

    # squeeze → (N,D)
    preds = np.squeeze(predictions)
    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    # Якщо batch‑axe лишилася
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]

    boxes, confs, ids = [], [], []

    if preds.shape[1] not in (5, 6):
        return boxes, confs, ids   # unsupported

    # heuristic: pixel coords if xmax > 2
    pixel_coords = preds[:,0:4].max() > 2

    for cx, cy, w, h, conf in preds:
        if conf < CONF_THRESHOLD:
            continue
        if not pixel_coords:
            # нормовані 0‑1 → до сітки
            cx, cy, w, h = cx*nw, cy*nh, w*nw, h*nh
        # відкат letterbox
        cx -= dw; cy -= dh
        cx /= scale; cy /= scale
        w  /= scale; h  /= scale

        x_min = int(cx - w/2)
        y_min = int(cy - h/2)
        if w>0 and h>0:
            boxes.append([x_min, y_min, int(w), int(h)])
            confs.append(float(conf))
            ids.append(0)

    if not boxes:
        return [],[],[]
    idx = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
    idx = idx.flatten() if len(idx)>0 else []
    boxes = [boxes[i] for i in idx]
    confs = [confs[i] for i in idx]
    ids   = [ids[i]   for i in idx]
    return boxes, confs, ids

# ───────────────────────── малювання ───────────────────────────────

def draw_boxes(frame, boxes, confs):
    for (x,y,w,h), c in zip(boxes, confs):
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{c:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,255,0),2)

# ───────────────────────── main ────────────────────────────────────

def main(opts):
    if opts.model_type=='onnx' and not ONNXRUNTIME_AVAILABLE:
        print('onnxruntime не встановлено'); return
    if opts.model_type=='pt' and not ULTRALYTICS_AVAILABLE:
        print('ultralytics не встановлено'); return
    if opts.model_type=='tflite' and not TENSORFLOW_AVAILABLE:
        print('tensorflow не встановлено'); return

    # ── завантаження моделі ───────────────────────────────────────
    model = None; sess=None; tfl=None; t_in=None; t_out=None
    input_size = (640,640)
    if opts.model_type=='pt':
        model = YOLO(opts.pt_path)
        input_size = (model.model.args['imgsz'],)*2 if hasattr(model,'model') else (640,640)
    elif opts.model_type=='onnx':
        sess = onnxruntime.InferenceSession(opts.onnx_path, providers=['CPUExecutionProvider'])
        shp  = sess.get_inputs()[0].shape  # [1,3,H,W]
        input_size = (shp[3], shp[2])
    else:
        tfl = tf.lite.Interpreter(model_path=opts.tflite_path, num_threads=opts.tflite_threads)
        tfl.allocate_tensors(); t_in=tfl.get_input_details()[0]; t_out=tfl.get_output_details()[0]
        shp  = t_in['shape']                # [1,3,H,W] or [1,H,W,3]
        if len(shp)==4 and shp[1]==3:
            input_size=(shp[3], shp[2])
        else:
            input_size=(shp[2], shp[1])

    print('Вхідний розмір сітки:', input_size)

    # ── відео ──────────────────────────────────────────────────────
    cap=cv2.VideoCapture(opts.video)
    if not cap.isOpened():
        print('Не вдалося відкрити відео'); return

    total_conf, det_cnt = 0.0, 0
    frame_id = 0
    last_det_frame = None

    while True:
        ret, frame = cap.read();
        if not ret: break
        frame_id+=1
        if opts.frame_skip>0 and (frame_id-1)% (opts.frame_skip+1):
            # пропускаємо, але можемо показувати попереднє
            if not opts.no_display and last_det_frame is not None:
                cv2.imshow('YOLO', last_det_frame); cv2.waitKey(1)
            continue

        blob, scale, dw, dh = preprocess_letterbox(frame, input_size)
        boxes=[]; confs=[]
        if model is not None:  # pt
            res=model(frame,verbose=False, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, classes=[TARGET_CLASS_ID])
            for r in res:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy());
                    boxes.append([x1,y1,x2-x1,y2-y1]); confs.append(float(b.conf))
        elif sess is not None: # onnx
            inp_name=sess.get_inputs()[0].name
            out=sess.run(None,{inp_name:blob})[0]
            boxes, confs,_= postprocess_yolo_output(out, frame.shape[:2], input_size,scale,dw,dh)
        else: # tflite
            inp = blob.transpose(0,2,3,1) if t_in['shape'][1]!=3 else blob  # NHWC
            # quant
            if t_in['dtype']==np.int8:
                scale_in, zp_in = t_in['quantization']
                inp = np.clip(inp/scale_in+zp_in,-128,127).astype(np.int8)
            tfl.set_tensor(t_in['index'], inp)
            tfl.invoke()
            out = tfl.get_tensor(t_out['index'])
            # dequant
            if t_out['dtype']==np.int8:
                s,z = t_out['quantization']; out = (out.astype(np.float32)-z)*s
            boxes, confs,_= postprocess_yolo_output(out, frame.shape[:2], input_size,scale,dw,dh)

        if confs:
            total_conf += sum(confs); det_cnt += len(confs)

        if not opts.no_display:
            vis = frame.copy(); draw_boxes(vis, boxes, confs); last_det_frame=vis
            cv2.imshow('YOLO', vis); if cv2.waitKey(1)&0xFF==ord('q'):break
        else:
            if frame_id%100==0:
                mean_c = total_conf/det_cnt if det_cnt else 0
                print(f"Кадр {frame_id}: детекцій={det_cnt}, середня впевненість={mean_c:.3f}")

    cap.release();
    if not opts.no_display: cv2.destroyAllWindows()
    mean_c = total_conf/det_cnt if det_cnt else 0
    print(f"\n=== Завершено. Загальна кількість детекцій: {det_cnt}; середня впевненість: {mean_c:.3f} ===")

# ───────────────────────── cli ─────────────────────────────────────
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True)
    p.add_argument('--model_type',required=True,choices=['pt','onnx','tflite'])
    p.add_argument('--pt_path'); p.add_argument('--onnx_path'); p.add_argument('--tflite_path')
    p.add_argument('--tflite_threads',type=int,default=4)
    p.add_argument('--frame_skip',type=int,default=0)
    p.add_argument('--no_display',action='store_true')
    opts=p.parse_args()
    main(opts)
