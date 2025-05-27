import cv2
import os
import platform

class CameraFeed:
    def __init__(self, fallback_video="test_video.mp4", fallback_image="test_image.jpg"):
        # Initialize flags and fallback resources
        self.is_connected = False
        self.cap = None
        self.fallback_video = fallback_video
        self.fallback_image = fallback_image
        self.fallback_frame = None
        self.available_cameras = {}
        self.current_camera = None

        # Scan for available cameras
        self.scan_cameras()

        # If no cameras are found, initialize with fallback
        if not self.available_cameras:
            print("No cameras found. Using fallback resources.")
            self.is_connected = False
            self._init_fallback()
        else:
            # Initialize with the first available camera by default
            self.current_camera = list(self.available_cameras.keys())[0]
            self._init_camera(self.current_camera)

    def scan_cameras(self):
        # Scan for all connected cameras
        self.available_cameras = {}

        # Check standard camera indices (webcams, USB cameras, virtual cameras like Iriun)
        for i in range(10):  # Check indices 0-9
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                self.available_cameras[f"Camera {i} (Index {i})"] = i
                cap.release()

        # On Linux, also check for GStreamer pipelines (e.g., /dev/videoX for USB cameras)
        if platform.system() == "Linux":
            for i in range(10):  # Check /dev/video0 to /dev/video9
                pipeline = (
                    f"v4l2src device=/dev/video{i} ! "
                    "video/x-raw, width=640, height=480, framerate=30/1 ! "
                    "videoconvert ! appsink"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    self.available_cameras[f"USB Camera /dev/video{i}"] = pipeline
                    cap.release()

        # Additional check for Windows wireless cameras (e.g., Iriun)
        # Iriun typically appears as a virtual camera, so it should be covered by the index scan
        # However, we can add a specific check if needed (e.g., using DirectShow on Windows)
        if platform.system() == "Windows":
            for i in range(10, 20):  # Check additional indices for virtual cameras
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.available_cameras[f"Virtual Camera {i} (Index {i})"] = i
                    cap.release()

    def _init_camera(self, camera_id):
        # Initialize the selected camera
        if self.cap is not None:
            self.cap.release()

        camera_source = self.available_cameras.get(camera_id)
        if isinstance(camera_source, str):  # GStreamer pipeline
            self.cap = cv2.VideoCapture(camera_source, cv2.CAP_GSTREAMER)
        else:  # Camera index
            if platform.system() == "Windows":
                self.cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(camera_source, cv2.CAP_ANY)

        if self.cap.isOpened():
            self.is_connected = True
            self.current_camera = camera_id
        else:
            print(f"Failed to initialize camera: {camera_id}")
            self.is_connected = False
            self._init_fallback()

    def _init_fallback(self):
        # Try to load a pre-recorded video for simulation
        if os.path.exists(self.fallback_video):
            self.cap = cv2.VideoCapture(self.fallback_video)
            if not self.cap.isOpened():
                print(f"Failed to load fallback video: {self.fallback_video}")
                self._load_fallback_image()
        else:
            print(f"Fallback video not found: {self.fallback_video}")
            self._load_fallback_image()

    def _load_fallback_image(self):
        # If video fails, load a static image as a last resort
        if os.path.exists(self.fallback_image):
            self.fallback_frame = cv2.imread(self.fallback_image)
            if self.fallback_frame is None:
                print(f"Failed to load fallback image: {self.fallback_image}")
                raise Exception("No fallback resources available")
        else:
            print(f"Fallback image not found: {self.fallback_image}")
            raise Exception("No fallback resources available")

    def get_frame(self):
        # Get a frame from the camera or fallback resource
        if self.is_connected:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        else:
            # If using a video, loop it
            if self.fallback_frame is None:
                ret, frame = self.cap.read()
                if not ret:  # Restart the video if it ends
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                return frame
            else:
                # Return the static image
                return self.fallback_frame.copy()

    def release(self):
        # Release the camera feed or video when done
        if self.cap is not None:
            self.cap.release()

    def check_connection(self):
        # Check if the camera is connected
        return self.is_connected

    def get_available_cameras(self):
        # Return the list of available cameras
        return list(self.available_cameras.keys())

    def set_camera(self, camera_id):
        # Set the active camera
        if camera_id in self.available_cameras:
            self._init_camera(camera_id)
            return True
        return False