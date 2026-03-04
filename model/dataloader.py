import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(
            data_dir, "images", "train"
        )  # Assuming images are in 'images/train'
        self.label_dir = os.path.join(
            data_dir, "labels", "train"
        )  # Assuming labels are in 'labels/train'
        self.img_names = sorted(os.listdir(self.img_dir))
        print(f"Found {len(self.img_names)} images in {self.img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load Image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load Labels (assuming YOLO format .txt)
        label_path = os.path.join(self.label_dir, img_name.rsplit(".", 1)[0] + ".txt")

        # Inside dataset.py -> class FaceDataset -> __getitem__

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = [float(x) for x in line.split()]
                    if len(parts) == 5:  # Ensure it's [class, x, y, w, h]
                        boxes.append(parts)

        # FIX: Ensure we always return a tensor of shape [5]
        if len(boxes) > 0:
            # Take only the first face found
            target = torch.tensor(boxes[0])
        else:
            # If no face is in the image, return zeros [class, x, y, w, h]
            target = torch.zeros(5)

        if self.transform:
            image = self.transform(image)

        return image, target
