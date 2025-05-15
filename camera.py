"""
camera.py
Адаптер, який підміняє cv2.VideoCapture та використовує libcamera (Picamera2)
на Raspberry Pi (основна камера або кілька камер). Якщо Picamera2 недоступна,
автоматично повертається до стандартного OpenCV.
"""

from typing import Tuple
import cv2

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False


class LibcameraCapture:
    """Інтерфейс максимально сумісний із cv2.VideoCapture."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fmt: str = "BGR888",
    ) -> None:
        self._using_picamera = _PICAMERA2_AVAILABLE
        if self._using_picamera:
            # --- libcamera шлях ---
            self._picam2 = Picamera2(camera_num=camera_index)
            video_cfg = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": fmt}
            )
            self._picam2.configure(video_cfg)
            self._picam2.start()
        else:
            # --- запасний варіант OpenCV ---
            self._cap = cv2.VideoCapture(camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # --------------------------------------------------------------------- #
    #               Методи, сумісні з cv2.VideoCapture                      #
    # --------------------------------------------------------------------- #
    def isOpened(self) -> bool:  # noqa: N802
        return True if self._using_picamera else self._cap.isOpened()

    def read(self) -> Tuple[bool, "np.ndarray"]:
        """
        :return: (ret, frame[BGR]). Формат кадру — BGR, як очікує OpenCV-код.
        """
        if self._using_picamera:
            try:
                frame = self._picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception as exc:  # pragma: no cover
                print(f"[camera] Picamera2 capture failed: {exc}")
                return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._using_picamera:
            self._picam2.stop()
        else:
            self._cap.release()
