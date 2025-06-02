import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import shutil
from camera_feed import CameraFeed
from object_detection import ObjectDetector
from robotic_arm import RoboticArm
from color_detection import ColorDetector

class WasteSortingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plastic Waste Sorting System")
        self.root.geometry("1280x800")

        # Color scheme for the GUI
        self.primary_color = "#F5F7FA"
        self.secondary_color = "#1A73E8"
        self.accent_color = "#34C759"
        self.text_color = "#202124"
        self.highlight_color = "#E8F0FE"
        self.error_color = "#D93025"

        # Initialize components
        try:
            self.camera = CameraFeed()
            self.camera_status = "Camera: Connected" if self.camera.check_connection() else "Camera: Not Connected (Using Fallback)"
        except Exception as e:
            self.camera = None
            self.camera_status = f"Camera: Failed to Initialize ({str(e)})"

        try:
            self.detector = ObjectDetector(model_path="HIT 400 YOLO.pt")
            # Get the class names from the YOLO model via ObjectDetector
            self.allowed_classes = self.detector.get_class_names()
        except Exception as e:
            print(f"Failed to initialize object detector: {e}")
            self.detector = None
            self.allowed_classes = []

        # Define class-specific colors for bounding boxes and labels (BGR format for OpenCV)
        # Use the YOLO model's class names
        self.class_colors = {}
        for cls_name in self.allowed_classes:
            if "PET" in cls_name:
                self.class_colors[cls_name] = (255, 0, 0)  # Blue for PET
            elif "HDPE" in cls_name:
                self.class_colors[cls_name] = (0, 255, 0)  # Green for HDPE
            elif "glass" in cls_name.lower():
                self.class_colors[cls_name] = (0, 0, 255)  # Red for glass
            else:
                self.class_colors[cls_name] = (128, 128, 128)  # Gray for unknown classes

        self.color_detector = ColorDetector()

        try:
            self.arm = RoboticArm()
            self.arm_status = "Arm: Connected" if self.arm.check_connection() else "Arm: Not Connected (Simulation Mode)"
        except Exception as e:
            self.arm = None
            self.arm_status = f"Arm: Failed to Initialize ({str(e)})"

        # Database
        self.database = {}
        self.database_dir = "database_images"
        os.makedirs(self.database_dir, exist_ok=True)

        # Operational settings
        self.arm_speed = 50
        self.operation_speed = 100
        self.selected_camera = None
        self.input_type = "High-Resolution Cameras"
        self.input_types = [
            "High-Resolution Cameras",
            "RGB-D Cameras (Depth Cameras)",
            "Hyperspectral Imaging Sensors",
            "Near-Infrared (NIR) Sensors",
            "Thermal Cameras",
            "LIDAR Sensors"
        ]
        self.arm_positions = {"x": 0, "y": 0, "z": 0}
        self.sorting_batches = {}

        # Variables for detection and sorting
        self.running_sorting = False  # Controls robotic arm sorting
        self.latest_detections = []  # Stores the most recent detection results
        self.latest_frame = None  # Stores the latest frame for display

        # Setup the GUI
        self.setup_gui()

        # Variables
        self.unrecognized_object = None
        self.unrecognized_bbox = None

        # Start the GUI update loop
        self.update_frame()

    def setup_gui(self):
        print("Setting up GUI...")
        self.button_style = ttk.Style()
        self.button_style.configure("TButton", font=("Helvetica", 14), padding=10, background=self.secondary_color, foreground=self.text_color)
        self.button_style.map("TButton",
                              background=[("active", self.accent_color), ("!disabled", self.secondary_color)],
                              foreground=[("active", self.primary_color)],
                              fieldbackground=[("active", self.highlight_color)])

        self.tab_style = ttk.Style()
        self.tab_style.configure("TNotebook", background=self.primary_color)
        self.tab_style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[20, 10], background=self.secondary_color, foreground=self.primary_color)
        self.tab_style.map("TNotebook.Tab",
                           background=[("selected", self.accent_color), ("active", self.highlight_color)],
                           foreground=[("selected", self.text_color), ("active", self.text_color)])

        self.main_frame = tk.Frame(self.root, bg=self.primary_color)
        self.main_frame.pack(fill="both", expand=True)

        header_frame = tk.Frame(self.main_frame, bg=self.primary_color)
        header_frame.pack(fill="x", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Status: Idle")
        self.status_label = tk.Label(header_frame, textvariable=self.status_var, font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color)
        self.status_label.pack(side="left", padx=10)

        self.camera_status_var = tk.StringVar(value=self.camera_status)
        self.camera_status_label = tk.Label(header_frame, textvariable=self.camera_status_var, font=("Helvetica", 14), bg=self.primary_color, fg=self.error_color if "Not Connected" in self.camera_status or "Failed" in self.camera_status else self.accent_color)
        self.camera_status_label.pack(side="left", padx=10)

        self.arm_status_var = tk.StringVar(value=self.arm_status)
        self.arm_status_label = tk.Label(header_frame, textvariable=self.arm_status_var, font=("Helvetica", 14), bg=self.primary_color, fg=self.error_color if "Not Connected" in self.arm_status or "Failed" in self.arm_status else self.accent_color)
        self.arm_status_label.pack(side="left", padx=10)

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.input_tab = tk.Frame(self.notebook, bg=self.primary_color)
        self.notebook.add(self.input_tab, text="Input")

        self.monitoring_tab = tk.Frame(self.notebook, bg=self.primary_color)
        self.notebook.add(self.monitoring_tab, text="Monitoring")

        self.settings_tab = tk.Frame(self.notebook, bg=self.primary_color)
        self.notebook.add(self.settings_tab, text="Settings")

        self.notebook.select(self.monitoring_tab)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.setup_input_tab()
        self.setup_monitoring_tab()
        self.setup_settings_tab()

    def on_tab_change(self, event):
        if True:  # transition_effects
            selected_tab = self.notebook.select()
            tab = self.notebook.nametowidget(selected_tab)
            tab.configure(bg=self.highlight_color)
            self.root.after(100, lambda: tab.configure(bg=self.primary_color))

    def setup_input_tab(self):
        print("Setting up Input tab...")
        input_container = tk.Frame(self.input_tab, bg=self.primary_color)
        input_container.pack(fill="both", expand=True, padx=20, pady=20)

        build_db_btn = ttk.Button(input_container, text="Build Database", command=self.build_database)
        build_db_btn.pack(pady=10, fill="x")

        camera_frame = tk.Frame(input_container, bg=self.primary_color)
        camera_frame.pack(pady=10, fill="x")
        tk.Label(camera_frame, text="Select Camera", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.available_cameras = self.camera.get_available_cameras() if self.camera else []
        self.selected_camera_var = tk.StringVar(value=self.available_cameras[0] if self.available_cameras else "No Cameras Available")
        camera_menu = ttk.OptionMenu(camera_frame, self.selected_camera_var, self.selected_camera_var.get(), *self.available_cameras, command=self.select_camera)
        camera_menu.pack(pady=5, fill="x")

        input_type_frame = tk.Frame(input_container, bg=self.primary_color)
        input_type_frame.pack(pady=10, fill="x")
        tk.Label(input_type_frame, text="Input Type", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.input_type_var = tk.StringVar(value=self.input_type)
        input_type_menu = ttk.OptionMenu(input_type_frame, self.input_type_var, self.input_type, *self.input_types, command=self.update_input_type)
        input_type_menu.pack(pady=5, fill="x")

    def setup_monitoring_tab(self):
        print("Setting up Monitoring tab...")

        main_container = tk.Frame(self.monitoring_tab, bg=self.primary_color)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_container, bg=self.primary_color)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_frame = tk.Frame(main_container, bg=self.primary_color)
        right_frame.grid(row=0, column=1, sticky="nsew")

        main_container.grid_columnconfigure(0, weight=3)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)

        camera_frame = tk.LabelFrame(left_frame, text="Camera Feed", font=("Helvetica", 14, "bold"), bg=self.primary_color, fg=self.text_color)
        camera_frame.pack(fill="both", expand=True)

        self.camera_feed_label = tk.Label(camera_frame, bg=self.primary_color)
        self.camera_feed_label.pack(pady=10, fill="both", expand=True)

        control_frame = tk.LabelFrame(right_frame, text="Controls", font=("Helvetica", 14, "bold"), bg=self.primary_color, fg=self.text_color)
        control_frame.pack(fill="both", expand=True)

        self.troubleshoot_var = tk.StringVar(value="Troubleshooting: No Issues")
        self.troubleshoot_label = tk.Label(control_frame, textvariable=self.troubleshoot_var, font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color)
        self.troubleshoot_label.pack(pady=10)

        select_arm_btn = ttk.Button(control_frame, text="Select Arm", command=self.select_arm)
        select_arm_btn.pack(pady=5, fill="x")

        sorting_batch_btn = ttk.Button(control_frame, text="Set Sorting Batch", command=self.set_sorting_batch)
        sorting_batch_btn.pack(pady=5, fill="x")

        database_btn = ttk.Button(control_frame, text="View Database", command=self.view_database)
        database_btn.pack(pady=5, fill="x")

        detect_btn = ttk.Button(control_frame, text="Detect", command=self.detect_objects)
        detect_btn.pack(pady=5, fill="x")

        button_frame = tk.Frame(control_frame, bg=self.primary_color)
        button_frame.pack(pady=10, fill="x")

        self.start_button = ttk.Button(button_frame, text="Start Sorting", command=self.start_sorting)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Sorting", command=self.stop_sorting, state="disabled")
        self.stop_button.pack(side="left", padx=5)

    def setup_settings_tab(self):
        print("Setting up Settings tab...")
        settings_container = tk.Frame(self.settings_tab, bg=self.primary_color)
        settings_container.pack(fill="both", expand=True, padx=20, pady=20)

        view_settings_btn = ttk.Button(settings_container, text="View Settings", command=self.view_settings)
        view_settings_btn.pack(pady=10, fill="x")

        speed_frame = tk.Frame(settings_container, bg=self.primary_color)
        speed_frame.pack(pady=10, fill="x")
        tk.Label(speed_frame, text="Operation Speed (RPM)", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.operation_speed_var = tk.DoubleVar(value=self.operation_speed)
        speed_slider = ttk.Scale(speed_frame, from_=50, to=500, orient="horizontal", variable=self.operation_speed_var, command=self.update_operation_speed)
        speed_slider.pack(pady=5, fill="x")
        self.operation_speed_entry = ttk.Entry(speed_frame, textvariable=self.operation_speed_var, font=("Helvetica", 14))
        self.operation_speed_entry.pack(pady=5, fill="x")

        arm_frame = tk.LabelFrame(settings_container, text="3D Arm Adjustment", font=("Helvetica", 14, "bold"), bg=self.primary_color, fg=self.text_color)
        arm_frame.pack(pady=10, fill="x")

        x_frame = tk.Frame(arm_frame, bg=self.primary_color)
        x_frame.pack(pady=5, fill="x")
        tk.Label(x_frame, text="X Position", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.x_pos_var = tk.DoubleVar(value=self.arm_positions["x"])
        x_slider = ttk.Scale(x_frame, from_=-100, to=100, orient="horizontal", variable=self.x_pos_var, command=lambda _: self.update_arm_position())
        x_slider.pack(pady=5, fill="x")

        y_frame = tk.Frame(arm_frame, bg=self.primary_color)
        y_frame.pack(pady=5, fill="x")
        tk.Label(y_frame, text="Y Position", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.y_pos_var = tk.DoubleVar(value=self.arm_positions["y"])
        y_slider = ttk.Scale(y_frame, from_=-100, to=100, orient="horizontal", variable=self.y_pos_var, command=lambda _: self.update_arm_position())
        y_slider.pack(pady=5, fill="x")

        z_frame = tk.Frame(arm_frame, bg=self.primary_color)
        z_frame.pack(pady=5, fill="x")
        tk.Label(z_frame, text="Z Position", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
        self.z_pos_var = tk.DoubleVar(value=self.arm_positions["z"])
        z_slider = ttk.Scale(z_frame, from_=-100, to=100, orient="horizontal", variable=self.z_pos_var, command=lambda _: self.update_arm_position())
        z_slider.pack(pady=5, fill="x")

    def build_database(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if files:
            group_name = tk.simpledialog.askstring("Input", "Enter group name (e.g., PET, HDPE):", parent=self.root)
            if group_name:
                group_dir = os.path.join(self.database_dir, group_name)
                os.makedirs(group_dir, exist_ok=True)
                for file in files:
                    dest_path = os.path.join(group_dir, os.path.basename(file))
                    shutil.copy(file, dest_path)
                if group_name not in self.database:
                    self.database[group_name] = []
                self.database[group_name].extend([os.path.join(group_dir, os.path.basename(file)) for file in files])
                messagebox.showinfo("Success", f"Added {len(files)} images to group {group_name}")

    def select_camera(self, value):
        if self.camera and value in self.available_cameras:
            self.camera.set_camera(value)
            self.selected_camera = value
            self.camera_status = "Camera: Connected" if self.camera.check_connection() else "Camera: Not Connected (Using Fallback)"
            self.camera_status_var.set(self.camera_status)
            self.camera_status_label.config(fg=self.accent_color if "Connected" in self.camera_status else self.error_color)
            messagebox.showinfo("Success", f"Selected camera: {value}")
        else:
            messagebox.showerror("Error", "Failed to select camera")

    def update_input_type(self, value):
        self.input_type = value
        messagebox.showinfo("Success", f"Input type set to: {value}")

    def select_arm(self):
        if not self.database:
            messagebox.showwarning("Warning", "No groups in database. Please build a database first.")
            return
        group = tk.simpledialog.askstring("Input", f"Select group to assign arm ({', '.join(self.database.keys())}):", parent=self.root)
        if group and group in self.database:
            messagebox.showinfo("Success", f"Arm assigned to sort group: {group}")
        else:
            messagebox.showerror("Error", "Invalid group name")

    def set_sorting_batch(self):
        if not self.database:
            messagebox.showwarning("Warning", "No groups in database. Please build a database first.")
            return
        group = tk.simpledialog.askstring("Input", f"Select group to set sorting location ({', '.join(self.database.keys())}):", parent=self.root)
        if group and group in self.database:
            bin_x = tk.simpledialog.askfloat("Input", "Enter bin X position:", parent=self.root)
            bin_y = tk.simpledialog.askfloat("Input", "Enter bin Y position:", parent=self.root)
            if bin_x is not None and bin_y is not None:
                self.sorting_batches[group] = (bin_x, bin_y)
                messagebox.showinfo("Success", f"Set sorting location for {group} to ({bin_x}, {bin_y})")
            else:
                messagebox.showerror("Error", "Invalid bin position")

    def view_database(self):
        print("View Database button clicked")
        if not self.database:
            messagebox.showinfo("Database", "Database is empty")
            return
        db_window = tk.Toplevel(self.root)
        db_window.title("Database")
        db_window.configure(bg=self.primary_color)
        for group, images in self.database.items():
            tk.Label(db_window, text=f"Group: {group}", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack()
            for img_path in images[:5]:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    label = tk.Label(db_window, image=img_tk, bg=self.primary_color)
                    label.image = img_tk
                    label.pack()
            ttk.Button(db_window, text=f"Delete {group}", command=lambda g=group: self.delete_group(g)).pack(pady=5)

    def delete_group(self, group):
        if group in self.database:
            shutil.rmtree(os.path.join(self.database_dir, group))
            del self.database[group]
            messagebox.showinfo("Success", f"Deleted group: {group}")

    def view_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("View Settings")
        settings_window.configure(bg=self.primary_color)

        tk.Label(settings_window, text="Transition Effects", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack(pady=5)
        self.transition_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_window, text="Enable", variable=self.transition_var, command=self.update_transition_effects).pack()

        tk.Label(settings_window, text="Text Size", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack(pady=5)
        self.text_size_var = tk.DoubleVar(value=14)
        ttk.Scale(settings_window, from_=12, to=20, orient="horizontal", variable=self.text_size_var, command=self.update_text_size).pack(pady=5)

    def update_transition_effects(self):
        pass  # Placeholder for transition effects

    def update_text_size(self, value):
        self.button_style.configure("TButton", font=("Helvetica", int(float(value))))
        self.tab_style.configure("TNotebook.Tab", font=("Helvetica", int(float(value))))

    def update_operation_speed(self, value):
        self.operation_speed = float(value)

    def update_arm_position(self):
        self.arm_positions["x"] = self.x_pos_var.get()
        self.arm_positions["y"] = self.y_pos_var.get()
        self.arm_positions["z"] = self.z_pos_var.get()
        if self.arm and self.arm.check_connection():
            self.arm.move_to(self.arm_positions["x"], self.arm_positions["y"], self.arm_positions["z"])

    def detect_objects(self):
        if self.camera is None:
            self.status_var.set("Status: Camera Not Initialized")
            return

        if self.detector is None:
            self.troubleshoot_var.set("Troubleshooting: Object Detector Not Initialized")
            self.status_var.set("Status: Error - Detector Not Initialized")
            return

        # Get the current frame
        frame = self.camera.get_frame()
        if frame is None:
            self.status_var.set("Status: No Frame Available")
            return

        # Step 1: Detect objects using YOLO model
        detection_result = self.detector.detect(frame)
        print(f"YOLO Detection Results: {detection_result}")

        # Step 2: Process detection results
        detections = []
        if not detection_result:  # Empty list
            print("No detections returned by YOLO model.")
        else:
            for detection in detection_result:
                if len(detection) == 3:  # Expected format: (cls_name, conf, bbox)
                    cls_name, conf, bbox = detection
                    detections.append((cls_name, conf, bbox))
                else:
                    print(f"Invalid detection format: {detection}")

        # Store the latest detections
        self.latest_detections = detections
        self.latest_frame = frame.copy()

        if not detections:
            self.troubleshoot_var.set("Troubleshooting: No valid objects detected")
        else:
            # Step 3: Process each detected object
            for idx, (cls_name, conf, bbox) in enumerate(detections):
                x1, y1, x2, y2 = map(int, bbox)

                # Step 4: Extract the ROI and detect the color
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_processed = self.color_detector.preprocess_roi(roi)
                    color = self.color_detector.detect_color(roi_processed[0])  # Pass the HSV ROI
                else:
                    color = "unknown"
                    print(f"Empty ROI for object {cls_name} (Index: {idx})")

                # Step 5: Store color in the detection tuple for sorting
                self.latest_detections[idx] = (cls_name, conf, bbox, color)

                # Step 6: Draw the colored bounding box
                box_color = self.class_colors.get(cls_name, (128, 128, 128))  # Default to gray if class not found
                cv2.rectangle(self.latest_frame, (x1, y1), (x2, y2), box_color, 2)

                # Step 7: Draw the label above the bounding box
                label = f"{cls_name} (Conf: {conf:.2f}) - {color}"
                label_y = max(15, y1 - 10)  # Ensure label is visible
                cv2.putText(self.latest_frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Step 8: Update troubleshooting status
            self.troubleshoot_var.set(f"Troubleshooting: Detected {len(detections)} objects")

        # Update the camera feed display with the processed frame
        self.update_display()

    def start_sorting(self):
        print("Starting sorting process...")
        if not self.latest_detections:
            messagebox.showwarning("Warning", "No objects detected. Please click 'Detect' first.")
            return

        self.running_sorting = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Status: Sorting")
        self.sort_objects()

    def stop_sorting(self):
        print("Stopping sorting process...")
        self.running_sorting = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Status: Idle")

    def sort_objects(self):
        if not self.running_sorting:
            return

        if not self.latest_detections:
            self.troubleshoot_var.set("Troubleshooting: No objects to sort")
            self.stop_sorting()
            return

        # Process each detected object for sorting
        for cls_name, conf, bbox, color in self.latest_detections:
            x1, y1, x2, y2 = map(int, bbox)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if self.arm is not None:
                self.arm.pick_and_sort(center_x, center_y, cls_name, color)
                print(f"Sorting {cls_name} (Color: {color}) at position ({center_x}, {center_y})")

        # After sorting, clear the detections to avoid repeated sorting
        self.latest_detections = []
        self.latest_frame = self.camera.get_frame()  # Reset to live frame
        self.update_display()

        # If sorting is still active, wait and check for new detections
        if self.running_sorting:
            self.root.after(1000, self.sort_objects)  # Check again after 1 second

    def update_display(self):
        # Update the camera feed display with the current frame
        frame = self.latest_frame if self.latest_frame is not None else self.camera.get_frame()
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_feed_label.imgtk = imgtk
        self.camera_feed_label.configure(image=imgtk)

    def update_frame(self):
        if self.camera is None:
            self.status_var.set("Status: Camera Not Initialized")
            self.root.after(10, self.update_frame)
            return

        # Always update the display with the latest frame if no detections are being shown
        if not self.latest_detections:
            self.latest_frame = self.camera.get_frame()
            self.update_display()

        self.root.after(10, self.update_frame)

    def handle_unrecognized_object(self, bbox, image):
        self.unrecognized_bbox = bbox
        self.unrecognized_object = image
        label_window = tk.Toplevel(self.root)
        label_window.title("Label Unrecognized Object")
        label_window.configure(bg=self.primary_color)

        tk.Label(label_window, text="Enter the label for this object:", font=("Helvetica", 14), bg=self.primary_color, fg=self.text_color).pack(pady=5)
        entry = ttk.Entry(label_window, font=("Helvetica", 14))
        entry.pack(pady=5)

        def submit_label():
            label = entry.get().strip()
            if label:
                self.detector.save_unrecognized_object(self.unrecognized_object, label)
                self.detector.retrain_model()
                messagebox.showinfo("Success", f"Object labeled as {label} and model updated.")
                label_window.destroy()
            else:
                messagebox.showerror("Error", "Please enter a valid label.")

        ttk.Button(label_window, text="Submit", command=submit_label).pack(pady=5)

    def on_closing(self):
        if self.camera is not None:
            self.camera.release()
        if self.arm is not None:
            self.arm.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteSortingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()