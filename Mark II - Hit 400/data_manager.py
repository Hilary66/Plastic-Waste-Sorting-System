import os

class DataManager:
    def __init__(self, dataset_dir="dataset/"):
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

    def append_data(self, image_path, label):
        # Append new data to the dataset
        annotation_path = os.path.join(self.dataset_dir, "labels.txt")
        with open(annotation_path, "a") as f:
            f.write(f"{image_path} {label}\n")
