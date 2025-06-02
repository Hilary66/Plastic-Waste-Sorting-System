# object_detection.py
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Use the class names defined in the YOLO model
        self.class_names = self.model.names  # Dictionary mapping class indices to names (e.g., {0: 'PET', 1: 'HDPE', ...})
        self.allowed_classes = list(self.class_names.values())  # Convert to list for filtering

    def detect(self, frame):
        # Run YOLO inference
        results = self.model(frame)
        detections = []

        # Process each detection
        for result in results:
            for box in result.boxes:
                # Get the class index and map it to the class name
                cls_index = int(box.cls)
                if cls_index < 0 or cls_index >= len(self.class_names):
                    print(f"Invalid class index {cls_index}. Skipping detection.")
                    continue
                cls_name = self.class_names[cls_index]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Since we're using the YOLO model's class names, all detected classes are allowed
                detections.append((cls_name, conf, (x1, y1, x2, y2)))

        if not detections:
            return []
        return detections

    def get_class_names(self):
        # Method to provide class names to other components
        return self.allowed_classes

    def save_unrecognized_object(self, image, label):
        # Placeholder for saving unrecognized objects
        pass

    def retrain_model(self):
        # Placeholder for retraining the model
        pass