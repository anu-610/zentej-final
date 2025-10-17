# File: dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class PairedFaceDataset(Dataset):
    """Custom Dataset class to load image pairs based on the CSV file."""
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img1_path = self.data_frame.iloc[idx, 0]
        img2_path = self.data_frame.iloc[idx, 1]
        identity_label = self.data_frame.iloc[idx, 2]
        forgery_label = self.data_frame.iloc[idx, 3]

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found. Skipping index {idx}.")
            return self.__getitem__((idx + 1) % len(self)) # Skip to next item

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'image1': img1,
            'image2': img2,
            'identity_label': torch.tensor(identity_label, dtype=torch.float),
            'forgery_label': torch.tensor(forgery_label, dtype=torch.float)
        }