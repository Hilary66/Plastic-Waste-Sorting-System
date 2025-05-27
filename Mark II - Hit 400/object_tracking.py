from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class ObjectTracker:
    def __init__(self):
        # Initialize the DeepSORT tracker
        self.tracker = DeepSort(max_age=30, nn_budget=100)

    def track(self, detections, frame):
        # Convert detections to the format required by DeepSORT
        # Format: [[x1, y1, w, h, conf], ...]
        deepsort_detections = []
        for cls_name, (x1, y1, x2, y2), conf in detections:
            w = x2 - x1
            h = y2 - y1
            deepsort_detections.append(([x1, y1, w, h], conf, cls_name))

        # Update the tracker with the new detections
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()  # Top-left, bottom-right coordinates
            cls_name = track.det_class
            tracked_objects.append((track_id, cls_name, bbox))
        return tracked_objects
