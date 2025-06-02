# color_detection.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorDetector:
    def __init__(self):
        """
        Initialize the ColorDetector with a predefined list of colors in RGB space.
        """
        # Predefined color list in RGB
        self.color_reference = {
            "White": (255, 255, 255),
            "Red": (255, 0, 0),
            "Green": (0, 128, 0),
            "Blue": (0, 0, 255),
            "Brown": (56, 28, 8),
            "Yellow": (255, 255, 0),
        }

        self.last_dominant_color = None
        self.last_roi_hash = None

    def preprocess_roi(self, roi):
        """
        Preprocess the ROI: normalize lighting and resize.
        """
        if roi.size == 0:
            return None, None

        # Resize ROI to a smaller size for faster processing
        roi = cv2.resize(roi, (50, 50))

        # Create a simple mask to ignore very dark areas
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 0, 50]), np.array([180, 255, 255]))
        return roi, mask

    def get_dominant_color(self, roi, mask):
        """
        Extract the dominant color from the ROI using K-Means clustering in RGB space.
        Cache the result to avoid recomputation for unchanged ROIs.
        """
        if roi is None or mask is None:
            return None

        # Compute a hash of the ROI to detect changes
        roi_hash = hash(roi.tobytes())
        if self.last_roi_hash == roi_hash and self.last_dominant_color is not None:
            return self.last_dominant_color

        pixels = roi.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        valid_pixels = pixels[mask_flat > 0]
        if len(valid_pixels) < 50:
            return None

        # Use a single cluster for faster processing
        kmeans = KMeans(n_clusters=1, random_state=0)
        kmeans.fit(valid_pixels)
        dominant_color = kmeans.cluster_centers_[0].astype(int)

        # Clamp dominant color to valid RGB range
        dominant_color = np.clip(dominant_color, 0, 255)

        # Cache the result
        self.last_dominant_color = dominant_color
        self.last_roi_hash = roi_hash
        return dominant_color

    def find_closest_color(self, rgb_color):
        """
        Find the closest color name to the given RGB color using Euclidean distance in RGB space.
        """
        if rgb_color is None:
            return "unknown"

        min_distance = float('inf')
        closest_color = "unknown"

        for color_name, rgb_ref in self.color_reference.items():
            diff = np.array(rgb_color) - np.array(rgb_ref)
            distance = np.sqrt(np.sum(diff**2))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color

    def detect_color(self, roi):
        """
        Detect the color of the ROI by finding the dominant color and matching it to the closest color.
        """
        if roi.size == 0:
            return "unknown"

        # Debug: Check for invalid pixel values in ROI
        if np.any(roi < 0) or np.any(roi > 255):
            print(f"Warning: ROI contains invalid pixel values: min {roi.min()}, max {roi.max()}")

        rgb_roi, mask = self.preprocess_roi(roi)
        if rgb_roi is None:
            return "unknown"

        dominant_color = self.get_dominant_color(rgb_roi, mask)
        if dominant_color is None:
            return "unknown"

        # Match to the closest color in RGB space
        color_name = self.find_closest_color(dominant_color)
        return color_name