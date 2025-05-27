import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        # Define HSV color ranges for various colors
        # Format: (lower_bound, upper_bound) in HSV space (H: 0-180, S: 0-255, V: 0-255)
        self.color_ranges = {
            "red": [(0, 50, 50), (10, 255, 255)],  # Red (first range)
            "red2": [(170, 50, 50), (180, 255, 255)],  # Red (second range due to hue wrapping)
            "green": [(35, 50, 50), (85, 255, 255)],  # Green
            "blue": [(100, 50, 50), (130, 255, 255)],  # Blue
            "clear": [(0, 0, 200), (180, 30, 255)],  # Clear (low saturation, high value)
            "yellow": [(20, 50, 50), (35, 255, 255)],  # Yellow
            "orange": [(10, 50, 50), (20, 255, 255)],  # Orange (between red and yellow)
            "white": [(0, 0, 200), (180, 30, 255)],  # White (low saturation, high value)
            "black": [(0, 0, 0), (180, 255, 50)],  # Black (low value)
            "purple": [(130, 50, 50), (160, 255, 255)],  # Purple
            "brown": [(10, 50, 20), (20, 255, 100)],  # Brown (dark orange tones)
        }

    def detect_color(self, roi):
        # Detect the dominant color in the given region of interest (ROI)
        # Convert the ROI from BGR to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # First, check for white and black, which rely more on saturation and value
        # Check for white: low saturation, high value
        white_mask = cv2.inRange(hsv_roi, self.color_ranges["white"][0], self.color_ranges["white"][1])
        white_ratio = np.sum(white_mask) / (roi.shape[0] * roi.shape[1] * 255)
        if white_ratio > 0.5:  # If more than 50% of the ROI matches the white range
            return "white"

        # Check for black: low value
        black_mask = cv2.inRange(hsv_roi, self.color_ranges["black"][0], self.color_ranges["black"][1])
        black_ratio = np.sum(black_mask) / (roi.shape[0] * roi.shape[1] * 255)
        if black_ratio > 0.5:  # If more than 50% of the ROI matches the black range
            return "black"

        # For other colors, create a mask to filter out low saturation and low value pixels
        mask = cv2.inRange(hsv_roi, (0, 30, 30), (180, 255, 255))

        # Compute the histogram of hues (H channel) for the masked region
        hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        dominant_hue = np.argmax(hist)  # Find the hue with the highest frequency

        # Initialize the detected color as "unknown"
        detected_color = "unknown"

        # Check which color range the dominant hue falls into
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name in ["white", "black", "clear"]:  # Skip white, black, and clear (already checked)
                continue

            lower_hue, lower_sat, lower_val = lower
            upper_hue, upper_sat, upper_val = upper

            # Special case for red, which wraps around the hue spectrum
            if color_name == "red2":
                if dominant_hue >= lower_hue:
                    detected_color = "red"
                    break
            else:
                if lower_hue <= dominant_hue <= upper_hue:
                    detected_color = color_name
                    break

        return detected_color

    def preprocess_roi(self, roi):
        # Preprocess the ROI to improve color detection accuracy
        # Apply a Gaussian blur to reduce noise
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        return roi
