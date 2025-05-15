# recognition/recognizer.py
import logging
import os
import cv2 # Для завантаження зображення, якщо detector цього не робить
from typing import Optional

# Припускаємо, що detector_yolo.py, ocr_easy.py та ocr_utils.py
# знаходяться в тій же директорії 'recognition' або правильно встановлені в PYTHONPATH
# Або, якщо вони в цій же директорії:
from .detector_yolo import Detector 
from .ocr_easy import EasyOcr
# Файл ocr_utils.py використовується всередині ocr_easy.py через `from recognition.utils import ocr_img_preprocess`
# Якщо ocr_utils.py тепер називається ocr_utils.py в цій же папці, то в ocr_easy.py
# імпорт має бути `from .ocr_utils import ocr_img_preprocess`

# Налаштування логування (можна використовувати централізоване з main.py або налаштувати тут)
# Якщо кожен модуль (detector, ocr) налаштовує свій логгер, це може бути достатньо.
# Ми будемо використовувати логгер, налаштований у PlateRecognizer.
logger = logging.getLogger(__name__)
# Щоб бачити логування з detector та ocr, якщо вони пишуть у стандартний логгер
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PlateRecognizer:
    def __init__(self, 
                 yolo_weights_path: str = "detection/models/25ep_best.pt", 
                 yolo_img_size: int = 640,
                 ocr_lang: list = ['en'], # Розгляньте ['en', 'uk', 'ru'] для українських номерів
                 ocr_allow_list: str = '0123456789АВСЕНІКМОРТХABCEHIKMOPTX', # Додав українські літери, що схожі
                 log_level: str = 'INFO',
                 log_dir: str = './logs/'): # Шлях до логів відносно кореня проекту
        """
        Ініціалізує розпізнавач номерних знаків.
        Завантажує необхідні моделі детекції та OCR.

        Args:
            yolo_weights_path (str): Шлях до файлу ваг моделі YOLOv7.
            yolo_img_size (int): Розмір зображення для YOLOv7.
            ocr_lang (list): Список мов для EasyOCR.
            ocr_allow_list (str): Дозволений список символів для EasyOCR.
            log_level (str): Рівень логування ('INFO', 'DEBUG').
            log_dir (str): Директорія для збереження лог-файлів.
        """
        self._logger = logging.getLogger(f"{__name__}.PlateRecognizer")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        # Тут можна додати обробник для логгера PlateRecognizer, якщо потрібно окремий файл
        
        self.models_loaded = False
        self.detector = None
        self.ocr = None

        # Перевірка існування директорії логів та її створення
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                self._logger.info(f"Log directory created: {log_dir}")
            except Exception as e:
                self._logger.error(f"Could not create log directory {log_dir}: {e}")
                # Можливо, логування в файл не буде працювати для detector/ocr
                # якщо вони очікують на існування цієї директорії
                log_dir = None # Відключити логування в файл для дочірніх класів, якщо директорія не створена

        self._load_models(yolo_weights_path, yolo_img_size, ocr_lang, ocr_allow_list, log_level, log_dir)

    def _load_models(self, yolo_weights_path, yolo_img_size, ocr_lang, ocr_allow_list, log_level, log_dir):
        """
        Завантажує моделі детекції (YOLOv7) та розпізнавання (EasyOCR).
        """
        self._logger.info("Attempting to load number plate recognition models...")
        try:
            # Ініціалізація детектора YOLOv7
            # Потрібно переконатися, що шляхи всередині Detector (особливо до yolov7 папки) коректні
            # відносно того, як запускається скрипт.
            # `sys.path.insert(0, yolo_folder_dir)` всередині Detector може потребувати уваги.
            # Краще, якщо yolo_folder_dir визначається на основі шляху до detector_yolo.py
            
            # Шлях до директорії detection відносно поточної директорії recognizer.py
            current_dir = os.path.dirname(os.path.abspath(__file__)) # .../auto-gate/recognition/
            # Припускаємо, що директорія detection знаходиться на рівень вище
            # detection_base_dir = os.path.join(os.path.dirname(current_dir), "detection")
            # yolo_weights_abs_path = os.path.join(os.path.dirname(current_dir), yolo_weights_path)


            # Перевірка існування файлу ваг
            # Важливо: yolo_weights_path має бути відносним до кореня проекту, звідки запускається main.py,
            # або абсолютним. Detector сам додає до sys.path папку yolov7.
            if not os.path.exists(yolo_weights_path):
                self._logger.error(f"YOLO weights file not found at: {yolo_weights_path}")
                raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights_path}")

            self.detector = Detector(model_weights=yolo_weights_path, 
                                     img_size=yolo_img_size, 
                                     log_level=log_level, 
                                     log_dir=log_dir)
            self._logger.info("YOLOv7 Detector loaded successfully.")

            # Ініціалізація EasyOCR
            self.ocr = EasyOcr(lang=ocr_lang, 
                               allow_list=ocr_allow_list, 
                               log_level=log_level, 
                               log_dir=log_dir)
            self._logger.info("EasyOCR loaded successfully.")
            
            self.models_loaded = True
            self._logger.info("All recognition models loaded and ready.")

        except Exception as e:
            self._logger.error(f"Failed to load one or more recognition models: {e}", exc_info=True)
            self.models_loaded = False


    def recognize_plate_from_image(self, image_path: str) -> Optional[str]:
        """
        Розпізнає номерний знак на зображенні, заданому шляхом.

        Args:
            image_path (str): Шлях до файлу зображення.

        Returns:
            Optional[str]: Рядок з розпізнаним номером (у верхньому регістрі)
                           або None, якщо номер не розпізнано або сталася помилка.
        """
        if not self.models_loaded:
            self._logger.error("Recognition models are not loaded. Cannot recognize plate.")
            return None

        if not os.path.exists(image_path):
            self._logger.error(f"Image file not found at: {image_path}")
            return None

        self._logger.info(f"Starting plate recognition for image: {image_path}")

        try:
            # 1. Завантаження зображення (detector.run приймає шлях до файлу або cv2 зображення)
            # Ми передамо шлях, detector.run сам завантажить через LoadImage
            # Якщо detector.run очікує саме cv2 зображення, тоді:
            # image_cv2 = cv2.imread(image_path)
            # if image_cv2 is None:
            #     self._logger.error(f"Failed to read image {image_path} using OpenCV.")
            #     return None
            # detect_result = self.detector.run(image_cv2) # Передати зображення
            
            # Згідно з detector.py, він може приймати шлях inp_image
            detect_result = self.detector.run(image_path) # inp_image може бути шляхом або cv2 зображенням

            # detect_result: {'file_name', 'orig_img', 'cropped_img', 'bbox', 'det_conf'}
            cropped_plate_img = detect_result.get('cropped_img')

            if cropped_plate_img is None:
                self._logger.info(f"No number plate detected by YOLOv7 in {image_path}.")
                return None
            
            self._logger.debug(f"Plate detected by YOLOv7. Detection confidence: {detect_result.get('det_conf')}")

            # 2. Розпізнавання символів (OCR) на обрізаному зображенні
            # ocr.run очікує словник {'cropped_img': img_obj, 'file_name': str}
            # cropped_img з детектора вже є RGB (згідно з `cropped_img = cropped_img[:, :, ::-1]`)
            # але ocr_img_preprocess перетворює на grayscale.
            ocr_input_dict = {
                'cropped_img': cropped_plate_img, # cropped_img з detector вже numpy array
                'file_name': detect_result.get('file_name', os.path.basename(image_path))
            }
            ocr_result = self.ocr.run(ocr_input_dict)
            
            # ocr_result: {'text': str|None, 'confid': float|None}
            recognized_text = ocr_result.get('text')
            ocr_confidence = ocr_result.get('confid')

            if recognized_text:
                self._logger.info(f"Successfully recognized plate: {recognized_text.upper()} with OCR confidence: {ocr_confidence}")
                # Тут можна додати валідацію формату номера, якщо потрібно
                # if not self.is_valid_plate_format(recognized_text.upper()):
                #     self._logger.warning(f"Recognized text '{recognized_text.upper()}' has invalid format.")
                #     return None
                return recognized_text.upper()
            else:
                self._logger.warning(f"OCR did not recognize text on the detected plate from {image_path}.")
                return None

        except Exception as e:
            self._logger.error(f"Error during plate recognition for {image_path}: {e}", exc_info=True)
            return None

    # Функція валідації може бути корисною, але її потрібно адаптувати під ваші формати
    # def is_valid_plate_format(self, plate_text: str) -> bool:
    #     # ... (реалізація з регулярними виразами)
    #     return True 


# --- Тестовий блок для recognition/recognizer.py ---
if __name__ == '__main__':
    # Цей тестовий блок запускатиметься з директорії auto-gate/recognition/
    # Потрібно, щоб шляхи до моделей та тестових зображень були відповідними.
    
    # Налаштовуємо логування для тестування
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Вивід у консоль
    
    logger.info("Тестування модуля recognizer.py...")

    # Відносні шляхи для тестування зсередини папки recognition/
    # Припускаємо, що папка detection/models/ знаходиться на рівень вище, потім в detection/
    # Тобто: ../detection/models/25ep_best.pt
    # Або, якщо запускати з кореня проекту: python -m recognition.recognizer
    # тоді шляхи мають бути відносно кореня проекту.
    
    # Для простоти тестування, припустимо, що ми запускаємо з кореня проекту:
    # python -m recognition.recognizer
    # І шляхи вказані відносно кореня проекту.
    
    # Або, якщо запускати python recognizer.py з папки recognition:
    # yolo_weights_test_path = "../detection/models/25ep_best.pt"
    # test_image_path = "../data/test/test1.jpg" # Потрібно мати такий файл для тесту
    # log_dir_test = "../logs/"

    # Варіант для запуску з кореня проекту (наприклад, `python auto-gate/recognition/recognizer.py`)
    # Визначимо корінь проекту, піднявшись на два рівні від __file__ (recognizer.py -> recognition -> auto-gate)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    yolo_weights_test_path = os.path.join(project_root, "detection/models/25ep_best.pt")
    # Створимо фейковий файл ваг, якщо його немає, щоб тест міг пройти без реальних ваг
    # (хоча Detector все одно спробує їх завантажити)
    os.makedirs(os.path.dirname(yolo_weights_test_path), exist_ok=True)
    if not os.path.exists(yolo_weights_test_path):
        logger.warning(f"Fake YOLO weights file created at {yolo_weights_test_path} for testing purposes. Real recognition will fail.")
        with open(yolo_weights_test_path, "w") as f: f.write("fake weights")

    log_dir_test = os.path.join(project_root, "logs")
    os.makedirs(log_dir_test, exist_ok=True)


    recognizer = PlateRecognizer(
        yolo_weights_path=yolo_weights_test_path,
        log_level="DEBUG", # Більше логів для тесту
        log_dir=log_dir_test
    )

    if not recognizer.models_loaded:
        logger.error("Моделі розпізнавання не були завантажені. Тестування неможливе.")
    else:
        # Шлях до тестового зображення (покладіть реальне зображення для тесту)
        # Припускаємо, що в auto-gate/ є папка data/test/ з зображеннями
        test_image_path_real = os.path.join(project_root, "data/test/plates/wp57yws.png") # З прикладу ocr.py
        
        # Створимо фейкове зображення, якщо реального немає, щоб тест міг пройти
        if not os.path.exists(test_image_path_real):
            os.makedirs(os.path.dirname(test_image_path_real), exist_ok=True)
            logger.warning(f"Test image {test_image_path_real} not found. Creating a dummy image.")
            dummy_img_content = cv2.imencode('.png', cv2.Mat(100, 400, cv2.CV_8UC3, (random.randint(0,255),random.randint(0,255),random.randint(0,255))))[1].tobytes()
            with open(test_image_path_real, "wb") as f:
                f.write(dummy_img_content)
        
        logger.info(f"\nТест 1: Розпізнавання з файлу '{test_image_path_real}'")
        recognized_number = recognizer.recognize_plate_from_image(test_image_path_real)

        if recognized_number:
            print(f"  Розпізнаний номерний знак: {recognized_number}")
        else:
            print("  Номерний знак не розпізнано або виникла помилка.")
    
    logger.info("Тестування модуля recognizer.py завершено.")