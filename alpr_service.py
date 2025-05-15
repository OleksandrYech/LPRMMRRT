"""
alpr_service.py
----------------
Робота з ultimateALPR-SDK та камерами Raspberry Pi (libcamera + Picamera2).

Інтерфейс:
    AlprService.detect_vehicle(frame: np.ndarray) -> dict | None
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
from picamera2 import Picamera2

# Doubango SDK
from ultimate_alpr_sdk import ultimateAlprSdk as ult

_LOG = logging.getLogger(__name__)


class AlprService:
    """Обгортує ініціалізацію UltimateALPR та дає простий detect_vehicle()."""

    def __init__(
        self,
        config_path: Path | str = Path(__file__).with_name("config.json"),
        license_token_data: str = "",
    ) -> None:
        self.config_path = Path(config_path)
        cfg_str = self._load_cfg()
        _LOG.info("Init UltimateALPR-SDK …")
        err = ult.UltAlprSdkEngine_init(cfg_str, license_token_data)
        if err:
            raise RuntimeError(f"UltimateALPR init failed: {err}")
        self.picam = self._init_camera()

    # ---------- public API ---------- #

    def detect_vehicle(self) -> Optional[Dict[str, str]]:
        """
        Знімає кадр з камери в’їзду та повертає:
            {
              "plate":  "AA1234BB",
              "make":   "Toyota",
              "model":  "Corolla",
              "color":  "White",
              "confidence": "93.4"
            }
        Якщо нічого не знайдено — None.
        """
        frame = self.picam.capture_array()
        h, w = frame.shape[:2]

        # SDK чекає байти у форматі BGR.
        result_json = ult.UltAlprSdkEngine_process(
            ult.ULTALPR_SDK_IMAGE_TYPE_BGR24,
            frame.ctypes.data,
            w,
            h,
            w * 3,
            1,  # 1 = доки не буде явного UltAlprSdkEngine_process_end()
        )
        res = json.loads(result_json)

        if res["plates"]:
            best = res["plates"][0]
            plate = best["text"].replace(" ", "")
            details = {
                "plate": plate,
                "make": best.get("vehicle", {}).get("make", ""),
                "model": best.get("vehicle", {}).get("model", ""),
                "color": best.get("vehicle", {}).get("color", ""),
                "confidence": f"{best['score']:.1f}",
            }
            _LOG.debug("ALPR hit: %s", details)
            return details
        return None

    # ---------- private helpers ---------- #

    def _load_cfg(self) -> str:
        default_cfg = {
            "debug_level": "warning",
            "num_threads": 4,
            "gpgpu_enabled": 0,
            "detect_roi": "0 0 100 100",
            "enable_make_model_recognizer": True,
            "enable_color_recognizer": True,
        }
        if self.config_path.exists():
            default_cfg.update(json.loads(self.config_path.read_text()))
        return json.dumps(default_cfg)

    @staticmethod
    def _init_camera() -> Picamera2:
        picam = Picamera2()
        preview = picam.create_still_configuration(
            main={"size": (1280, 720), "format": "BGR888"}
        )
        picam.configure(preview)
        picam.start()
        return picam

    # ---------- cleanup ---------- #

    def __del__(self) -> None:
        try:
            ult.UltAlprSdkEngine_deInit()
        except Exception:  # noqa: BLE001
            pass
