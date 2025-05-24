def parse_tflite_out(interp, out_det, thr: float, names,
                     img_w: int = 320, iou_thr: float = 0.3):
    """
    Повертає [(x_center_px, char, conf), …] з 1-D IoU-NMS.
    img_w  – ширина входу (у вас 320).
    iou_thr – поріг перетину відрізків по X, за якого бокс відкидається.
    """

    # ---- допоміжні -------------------------------------------------------
    def dequant(arr: np.ndarray, od: dict) -> np.ndarray:
        if arr.dtype == np.float32:
            return arr
        qp = od["quantization_parameters"]
        return (arr.astype(np.float32) - qp["zero_points"]) * qp["scales"]

    def iou1d(a1, a2):
        """IoU для відрізків [x1,x2]."""
        l = max(a1[0], a2[0])
        r = min(a1[1], a2[1])
        inter = max(0.0, r - l)
        union = (a1[1] - a1[0]) + (a2[1] - a2[0]) - inter
        return inter / union if union > 0 else 0.0

    # ---- збираємо сирі детекції ------------------------------------------
    raw_det = []   # (x1,x2,char,conf)

    # 1) 4-тензорний вихід (у пікселях, NMS уже виконано)
    if len(out_det) == 4 and out_det[0]["shape"][-1] == 4:
        boxes   = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])[0]
        scores  = dequant(interp.get_tensor(out_det[1]["index"]), out_det[1])[0]
        classes = dequant(interp.get_tensor(out_det[2]["index"]), out_det[2])[0]
        n = int(interp.get_tensor(out_det[3]["index"])[0])

        for j in range(n):
            conf = float(scores[j])
            if conf < thr:                   continue
            cid = int(classes[j])
            if cid >= len(names):            continue
            x1, _, x2, _ = boxes[j]
            raw_det.append((x1, x2, names[cid], conf))

    # 2) один великий тензор
    else:
        raw = dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if raw.ndim != 3:
            return []

        attrs = raw.shape[2]

        # 2.a «N×6»
        if attrs == 6:
            for x1, _, x2, _, conf, cid in raw[0]:
                conf = float(conf); cid = int(cid)
                if conf < thr or cid >= len(names):  continue
                raw_det.append((x1, x2, names[cid], conf))

        # 2.b «N×(5+nc)» сирий YOLO
        elif attrs >= 5 + len(names):
            sigm = lambda x: 1.0 / (1.0 + math.exp(-x))
            for row in raw[0]:
                obj = sigm(float(row[4]))
                if obj < 1e-6:                continue
                logits = row[5 : 5 + len(names)]
                cid = int(np.argmax(logits))
                cls_prob = sigm(float(logits[cid]))
                conf = obj * cls_prob
                if conf < thr:                continue
                if cid >= len(names):         continue
                cx  = float(row[0]) * img_w
                w   = float(row[2]) * img_w
                x1  = cx - w / 2.0
                x2  = cx + w / 2.0
                raw_det.append((x1, x2, names[cid], conf))

    if not raw_det:
        return []

    # ---- 1-D IoU-NMS ------------------------------------------------------
    # сортуємо за conf ↓
    raw_det.sort(key=lambda d: d[3], reverse=True)
    keep = []

    for cand in raw_det:
        x1, x2, ch, cf = cand
        if all(iou1d((x1, x2), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    # сортуємо залишені символи за центром X
    keep.sort(key=lambda d: (d[0] + d[1]) / 2.0)
    return [((k[0] + k[1]) / 2.0, k[2], k[3]) for k in keep]