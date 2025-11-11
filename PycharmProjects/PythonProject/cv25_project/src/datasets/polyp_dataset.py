import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class PolypDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.images = []
        self.labels = []

        for label_dir in ["polyp", "non_polyp"]:
            full_path = os.path.join(root_dir, label_dir)
            for img_name in os.listdir(full_path):
                if img_name.endswith(".jpg") or img_name.endswith(".png"):
                    self.images.append(os.path.join(full_path, img_name))
                    self.labels.append(0 if label_dir == "non_polyp" else 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
