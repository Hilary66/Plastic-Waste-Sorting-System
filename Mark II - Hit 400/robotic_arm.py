import serial
import time

class RoboticArm:
    def __init__(self, port="COM3", baudrate=9600):
        # Initialize flags and bin positions
        self.is_connected = False
        self.ser = None
        self.bins = {
            "PET_clear": (100, 200),  # Clear PET bin
            "PET_blue": (150, 200),   # Blue PET bin
            "PET_green": (200, 200),  # Green PET bin
            "PET_white": (250, 200),  # White PET bin
            "PET_black": (300, 200),  # Black PET bin
            "PET_orange": (350, 200), # Orange PET bin
            "PET_purple": (400, 200), # Purple PET bin
            "PET_brown": (450, 200),  # Brown PET bin
            "HDPE_clear": (500, 200), # Clear HDPE bin
            "HDPE_blue": (550, 200),  # Blue HDPE bin
            "HDPE_green": (600, 200), # Green HDPE bin
            "HDPE_white": (650, 200), # White HDPE bin
            "HDPE_black": (700, 200), # Black HDPE bin
            "HDPE_orange": (750, 200),# Orange HDPE bin
            "HDPE_purple": (800, 200),# Purple HDPE bin
            "HDPE_brown": (850, 200), # Brown HDPE bin
            "unknown": (900, 200),    # Default bin for unknown objects
        }

        # Try to initialize the serial connection to the robotic arm
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.is_connected = True
            print("Robotic arm connected successfully")
        except Exception as e:
            print(f"Robotic arm connection failed: {e}")
            self.is_connected = False

    def move_to(self, x, y, z):
        # Send command to move the arm if connected
        if self.is_connected:
            command = f"MOVE {x} {y} {z}\n"
            self.ser.write(command.encode())
            time.sleep(1)  # Wait for the arm to move
        else:
            print(f"Simulating move to ({x}, {y}, {z})")

    def pick_and_sort(self, x, y, cls_name, color):
        # Pick and sort the object if the arm is connected
        if self.is_connected:
            self.move_to(x, y, 0)  # Move to the object position
            self.move_to(x, y, -10)  # Lower the arm to pick
            self.ser.write(b"GRIP 1\n")  # Close the gripper
            time.sleep(0.5)
            self.move_to(x, y, 0)  # Lift the object

            # Determine the bin based on class and color
            bin_key = f"{cls_name}_{color}" if color != "unknown" else "unknown"
            bin_x, bin_y = self.bins.get(bin_key, self.bins["unknown"])
            self.move_to(bin_x, bin_y, 0)  # Move to the bin
            self.move_to(bin_x, bin_y, -10)  # Lower the arm
            self.ser.write(b"GRIP 0\n")  # Release the object
            time.sleep(0.5)
            self.move_to(bin_x, bin_y, 0)  # Lift the arm back up
        else:
            bin_key = f"{cls_name}_{color}" if color != "unknown" else "unknown"
            bin_x, bin_y = self.bins.get(bin_key, self.bins["unknown"])
            print(f"Simulating pick and sort: {cls_name} ({color}) to bin ({bin_x}, {bin_y})")

    def close(self):
        # Close the serial connection if it exists
        if self.is_connected and self.ser is not None:
            self.ser.close()

    def check_connection(self):
        # Check if the robotic arm is connected
        return self.is_connected