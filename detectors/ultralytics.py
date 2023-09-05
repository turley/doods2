import cv2
import odrpc
import logging
from config import DoodsDetectorConfig
from ultralytics import YOLO

class Ultralytics:
    def __init__(self, config: DoodsDetectorConfig):
        self.config = odrpc.Detector(**{
            'name': config.name,
            'type': 'ultralytics',
            'labels': [],
            'model': config.modelFile,
        })
        self.logger = logging.getLogger("doods.ultralytics."+config.name)
        self.ultralytics_model = YOLO(config.modelFile)
        if isinstance(self.ultralytics_model.names, dict):
            self.labels = list(self.ultralytics_model.names.values())
        else:
            self.labels = self.ultralytics_model.names
        self.config.labels = self.labels
        self.fullImageSize = config.fullImageSize
        self.halfPrecision = config.halfPrecision


    def round_to_multiple(self, number, multiple):
        return multiple * round(number / multiple)

    def detect(self, image):
        (height, width, colors) = image.shape
        largest_dimension = self.round_to_multiple(max(width, height), 32) if self.fullImageSize else 640

        # For some reason, we get bogus results on old AMD GPUs unless half precision is used
        results = self.ultralytics_model.predict(image, imgsz=largest_dimension, half=self.halfPrecision)

        ret = odrpc.DetectResponse()
        for box in results[0].boxes:
            detection = odrpc.Detection()
            xyxy = box.xyxy[0].tolist()
            (detection.top, detection.left, detection.bottom, detection.right) = (xyxy[1]/height, xyxy[0]/width, xyxy[3]/height, xyxy[2]/width)
            detection.confidence = box.conf[0].item() * 100.0
            detection.label = self.ultralytics_model.names[int(box.cls[0].item())]
            ret.detections.append(detection)
    
        return ret
