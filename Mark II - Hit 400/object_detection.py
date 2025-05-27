import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path="HIT 400 YOLO.pt"):
        # Initialize the YOLO model
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the YOLO model
        try:
            self.model = YOLO(self.model_path)
            print(f"Successfully loaded YOLO model from {self.model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise Exception(f"Failed to load YOLO model from {self.model_path}")

        # Placeholder for dataset directory (for retraining)
        self.dataset_dir = "dataset"

    def detect(self, frame):
        # Perform object detection on the given frame
        try:
            # Convert frame to the format expected by YOLO
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Frame shape: {img.shape}")

            # Run inference
            results = self.model(img)

            # Process detection results
            detected_objects = []
            for result in results:
                if result.boxes is None:
                    print("No objects detected in this frame")
                    continue

                for box in result.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id] if class_id < len(self.model.names) else "unknown"

                    # Filter detections based on confidence threshold
                    if confidence < 0.5:
                        continue

                    # Format: (class_name, bounding_box, confidence)
                    detected_objects.append((class_name, (x1, y1, x2, y2), confidence))

            if not detected_objects:
                print("No objects detected after filtering")
                # If no known objects are detected, check for potential new objects
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self.model.names[class_id] if class_id < len(self.model.names) else "unknown"
                        if class_name == "unknown" or confidence < 0.5:
                            # Return the first unknown object for labeling
                            roi = frame[y1:y2, x1:x2]
                            return ("unknown", (x1, y1, x2, y2), roi)

            print(f"Detected {len(detected_objects)} objects: {[obj[0] for obj in detected_objects]}")
            return detected_objects

        except Exception as e:
            print(f"Error during object detection: {e}")
            return []

    def save_unrecognized_object(self, image, label):
        # Save the unrecognized object image for retraining
        os.makedirs(self.dataset_dir, exist_ok=True)
        label_dir = os.path.join(self.dataset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f"unrecognized_{len(os.listdir(label_dir))}.jpg")
        cv2.imwrite(image_path, image)
        print(f"Saved unrecognized object as {image_path}")

    def retrain_model(self):
        # Placeholder for retraining the YOLO model with new data
        try:
            print("Retraining YOLO model with new data...")
            self.model.train(data=self.dataset_dir, epochs=10, imgsz=640)
            self.model.save("custom_yolo11.pt")
            print("Model retrained and saved as custom_yolo11.pt")
        except Exception as e:
            print(f"Error during retraining: {e}")