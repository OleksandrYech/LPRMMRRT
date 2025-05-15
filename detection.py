"""
ALPR-обгортка довкола Doubango UltimateALPR-SDK.
Перетворює кадр BGR ➜ I420 (bytes) та віддає в движок.
"""

import json
from pathlib import Path

import cv2
import ultimateAlprSdk

import config

# ─── Ініціалізація движка (Lazy Singleton) ────────────────────────────────────
_engine = ultimateAlprSdk.UltAlprSdkEngine()
if not _engine.isInit():
    params = {
        "debug_level": "warning",
        "charset": "latin",
        "pyramidal_search_enabled": "true",
        # додавайте інші параметри за потреби
    }
    _engine.init(json.dumps(params), config.ALPR_SDK_LIB_PATH)


def recognize_bgr(frame):
    """
    Повертає dict із результатами розпізнавання або None, якщо номерів немає.
    """
    height, width = frame.shape[:2]

    # OpenCV → I420 bytes (саме bytes, а не numpy-view — так стабільніше на armv7l)
    yuv_bytes = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes()
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    y_plane = yuv_bytes[0:y_size]
    u_plane = yuv_bytes[y_size:y_size + uv_size]
    v_plane = yuv_bytes[y_size + uv_size:y_size + 2 * uv_size]

    result_json = _engine.process(
        ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_YUV420P,
        y_plane, u_plane, v_plane,
        width, height,
        0, 0, 0  # strides (0 = packed)
    )

    result = json.loads(result_json)
    return result if result.get("plates") else None


def __del__():
    """Граційне звільнення ресурсів при завершенні роботи."""
    if _engine.isInit():
        _engine.deInit()
