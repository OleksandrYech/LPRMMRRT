# camera.py
import time
import logging
from picamera2 import Picamera2, Preview
from libcamera import Transform

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Очікувані частини назв моделей сенсорів для ідентифікації камер
DEFAULT_ENTRY_CAM_MODEL_SUBSTRING = 'imx708' # Для Camera Module 3
DEFAULT_EXIT_CAM_MODEL_SUBSTRING = 'imx219'  # Для Camera Module 2

def get_camera_ids(entry_cam_model_sub=DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                   exit_cam_model_sub=DEFAULT_EXIT_CAM_MODEL_SUBSTRING):
    """
    Знаходить індекси (camera_num) для камери в'їзду та виїзду на основі
    підрядків у назвах їх моделей сенсорів.

    Args:
        entry_cam_model_sub (str): Підрядок для ідентифікації моделі камери в'їзду (напр., 'imx708').
        exit_cam_model_sub (str): Підрядок для ідентифікації моделі камери виїзду (напр., 'imx219').

    Returns:
        dict: Словник з ключами "entry" та "exit" та відповідними індексами камер (camera_num),
              або None для відповідного ключа, якщо камеру не знайдено.
              Приклад: {"entry": 0, "exit": 1}
    """
    camera_ids = {"entry": None, "exit": None}
    try:
        cameras_info = Picamera2.global_camera_info()
        if not cameras_info:
            logger.warning("No cameras found by Picamera2.global_camera_info().")
            return camera_ids
        
        logger.info(f"Available cameras detected: {cameras_info}")

        for i, info in enumerate(cameras_info):
            model = info.get("Model", "").lower()
            cam_num = info.get("Num", i) # Використовуємо 'Num' якщо є, інакше індекс 'i'

            if entry_cam_model_sub in model and camera_ids["entry"] is None:
                camera_ids["entry"] = cam_num
                logger.info(f"Entry camera (model substring '{entry_cam_model_sub}') found: {info.get('Location', '')} {model} (ID: {cam_num})")
            elif exit_cam_model_sub in model and camera_ids["exit"] is None:
                camera_ids["exit"] = cam_num
                logger.info(f"Exit camera (model substring '{exit_cam_model_sub}') found: {info.get('Location', '')} {model} (ID: {cam_num})")
        
        if camera_ids["entry"] is None:
            logger.warning(f"Entry camera (model substring '{entry_cam_model_sub}') NOT found.")
        if camera_ids["exit"] is None:
            logger.warning(f"Exit camera (model substring '{exit_cam_model_sub}') NOT found.")
            
    except Exception as e:
        logger.error(f"Error while trying to get camera IDs: {e}")
    return camera_ids


class CameraController:
    """
    Клас для керування камерою Raspberry Pi за допомогою Picamera2.
    """
    def __init__(self, camera_id, camera_name="UnnamedCamera", capture_resolution=(1920, 1080), hflip=False, vflip=False):
        """
        Ініціалізує конкретну камеру.

        Args:
            camera_id (int): Індекс камери (camera_num), яку потрібно використовувати.
            camera_name (str): Описова назва камери (напр., "EntryCamera_CM3").
            capture_resolution (tuple): Роздільна здатність для захоплення (ширина, висота).
            hflip (bool): Віддзеркалити по горизонталі.
            vflip (bool): Віддзеркалити по вертикалі.
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.capture_resolution = capture_resolution
        self.picam2 = None
        self._logger = logging.getLogger(f"{__name__}.{self.camera_name}_ID{self.camera_id}")

        try:
            self.picam2 = Picamera2(camera_num=self.camera_id)
            cam_model_info = self.picam2.camera_properties.get('Model', 'Unknown Model')
            self._logger.info(f"Camera '{self.camera_name}' (ID: {self.camera_id}, Model: {cam_model_info}) initialized.")

            transform = Transform(hflip=hflip, vflip=vflip)
            config = self.picam2.create_still_configuration(
                main={"size": self.capture_resolution},
                lores={"size": (640, 480)},
                display="lores",
                transform=transform
            )
            self.picam2.configure(config)
            self._logger.info(f"Configured for still capture with resolution {self.capture_resolution}, transform: hflip={hflip}, vflip={vflip}.")
            
            self.picam2.start()
            self._logger.info("Camera started.")

            self._logger.info("Allowing 2-3 seconds for camera to adjust...")
            time.sleep(3)
            self._logger.info("Camera ready.")

        except Exception as e:
            self._logger.error(f"Failed to initialize or start: {e}")
            if self.picam2:
                try:
                    self.picam2.close()
                except Exception as close_e:
                    self._logger.error(f"Error closing camera during init failure: {close_e}")
            self.picam2 = None

    def capture_image(self, output_path="captured_image.jpg"):
        if not self.picam2 or not self.picam2.started:
            self._logger.error("Camera not initialized or not started. Cannot capture image.")
            return None
        try:
            self._logger.info(f"Attempting to capture image to {output_path}...")
            metadata = self.picam2.capture_file(output_path)
            self._logger.info(f"Image captured to {output_path}. Metadata: {metadata}")
            return output_path
        except Exception as e:
            self._logger.error(f"Failed to capture image to {output_path}: {e}")
            return None

    def capture_array(self, array_name="main"):
        if not self.picam2 or not self.picam2.started:
            self._logger.error("Camera not initialized or not started. Cannot capture array.")
            return None
        try:
            self._logger.info(f"Attempting to capture array from stream '{array_name}'...")
            image_array = self.picam2.capture_array(array_name)
            self._logger.info(f"Array captured from stream '{array_name}' with shape {image_array.shape}.")
            return image_array
        except Exception as e:
            self._logger.error(f"Failed to capture array from stream '{array_name}': {e}")
            return None
            
    def start_preview(self, x=100, y=100, width=640, height=480):
        if not self.picam2:
            self._logger.error("Camera not initialized. Cannot start preview.")
            return
        try:
            self.picam2.start_preview(Preview.DRM, x=x, y=y, width=width, height=height)
            self._logger.info(f"Preview started. Pos: ({x},{y}), Size: ({width}x{height})")
        except Exception as e:
            self._logger.error(f"Failed to start preview: {e}.")

    def stop_preview(self):
        if not self.picam2: return
        try:
            self.picam2.stop_preview()
            self._logger.info("Preview stopped.")
        except Exception as e:
            self._logger.error(f"Error stopping preview: {e}")

    def close(self):
        if self.picam2:
            try:
                if self.picam2.started:
                    self.picam2.stop_preview()
                    self.picam2.stop()
                    self._logger.info("Camera stopped.")
                self.picam2.close()
                self._logger.info("Camera resources released.")
                self.picam2 = None
            except Exception as e:
                self._logger.error(f"Error closing camera: {e}")

    def __del__(self):
        self.close()

# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    logger.info("Тестування модуля camera.py з двома камерами...")

    # Отримуємо індекси для камери в'їзду (CM3) та виїзду (CM2)
    camera_indices = get_camera_ids()
    
    entry_camera_controller = None
    exit_camera_controller = None

    # --- Ініціалізація та тест камери в'їзду (Camera Module 3) ---
    if camera_indices["entry"] is not None:
        logger.info(f"\n--- Тестування камери В'ЇЗДУ (ID: {camera_indices['entry']}) ---")
        entry_camera_controller = CameraController(
            camera_id=camera_indices["entry"],
            camera_name="EntryCamera_CM3",
            capture_resolution=(1920, 1080) # FullHD для CM3
        )
        if entry_camera_controller.picam2:
            try:
                output_file_entry = "test_capture_entry_cm3.jpg"
                # entry_camera_controller.start_preview(x=50, y=50, width=320, height=240) # Маленьке прев'ю
                # time.sleep(2)
                captured_entry = entry_camera_controller.capture_image(output_file_entry)
                if captured_entry:
                    logger.info(f"Камера В'ЇЗДУ: Зображення збережено як {captured_entry}")
                else:
                    logger.error("Камера В'ЇЗДУ: Не вдалося захопити зображення.")
                # entry_camera_controller.stop_preview()
            finally:
                entry_camera_controller.close()
        else:
            logger.error(f"Камера В'ЇЗДУ (ID: {camera_indices['entry']}) не ініціалізована.")
    else:
        logger.warning("Камера В'ЇЗДУ (Camera Module 3) не знайдена або не вдалося отримати її ID.")

    # --- Ініціалізація та тест камери виїзду (Camera Module 2) ---
    if camera_indices["exit"] is not None:
        logger.info(f"\n--- Тестування камери ВИЇЗДУ (ID: {camera_indices['exit']}) ---")
        # Camera Module 2 може мати інші оптимальні параметри або обмеження
        exit_camera_controller = CameraController(
            camera_id=camera_indices["exit"],
            camera_name="ExitCamera_CM2",
            capture_resolution=(1280, 720) # HD для CM2, або (1920, 1080) якщо підтримує добре
        )
        if exit_camera_controller.picam2:
            try:
                output_file_exit = "test_capture_exit_cm2.jpg"
                # exit_camera_controller.start_preview(x=400, y=50, width=320, height=240) # Прев'ю в іншому місці
                # time.sleep(2)
                captured_exit = exit_camera_controller.capture_image(output_file_exit)
                if captured_exit:
                    logger.info(f"Камера ВИЇЗДУ: Зображення збережено як {captured_exit}")
                else:
                    logger.error("Камера ВИЇЗДУ: Не вдалося захопити зображення.")
                # exit_camera_controller.stop_preview()
            finally:
                exit_camera_controller.close()
        else:
            logger.error(f"Камера ВИЇЗДУ (ID: {camera_indices['exit']}) не ініціалізована.")
    else:
        logger.warning("Камера ВИЇЗДУ (Camera Module 2) не знайдена або не вдалося отримати її ID.")

    logger.info("\nТестування модуля camera.py завершено.")