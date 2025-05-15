"""
detection.py
Розпізнавання номерного знаку та атрибутів транспортного засобу
за допомогою UltimateALPR‑SDK (https://github.com/DoubangoTelecom/ultimateALPR-SDK)
"""

from __future__ import annotations
import cv2
import json
import numpy as np
import ultimateAlprSdk
from typing import Tuple
import config

# ініціалізація движка
_INIT_RES = ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(config.ALPR_JSON_CONFIG))
if not _INIT_RES.isOK():
    raise RuntimeError(f"ALPR init failed: {_INIT_RES.phrase()}")

_IMG_TYPE = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24

def _process(frame: np.ndarray) -> dict:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    result = ultimateAlprSdk.UltAlprSdkEngine_process(
        _IMG_TYPE,
        rgb.tobytes(),
        w,
        h,
        w * 3,
        1
    )
    if not result.isOK():
        raise RuntimeError("ALPR process failed: " + result.phrase())
    return json.loads(result.json())

def detect_plate_and_vehicle(frame: np.ndarray) -> Tuple[str|None, str|None, str|None, str|None]:
    data = _process(frame)
    plates = data.get('plates', [])
    if not plates:
        return None, None, None, None
    best = plates[0]
    plate = best.get('plate')
    vehicle = best.get('vehicle', {})
    return (
        plate or None,
        vehicle.get('make') or None,
        vehicle.get('model') or None,
        vehicle.get('color') or None
    )

def detect_vehicle_exit(frame: np.ndarray, min_contour_area: int = 5000) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > min_contour_area for c in contours)

def deinit() -> None:
    res = ultimateAlprSdk.UltAlprSdkEngine_deInit()
    if not res.isOK():
        print("ALPR DeInit failed: " + res.phrase())
