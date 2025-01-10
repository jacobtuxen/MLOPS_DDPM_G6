import glob
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class Pokemon(Dataset):
    """Pokémon sprite dataset."""

    def __init__(self, raw_data_path: Path, transform: Any = None) -> None:
        self.data_path = raw_data_path
        self.transform = transform

        self.images = torch.load(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image

    def preprocess(self, raw_data_path: Path, preprocessed_path: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        os.makedirs(preprocessed_path, exist_ok=True)

        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        images = []

        for raw_path in glob.glob(f"{raw_data_path}/*.jpg"):
            with Image.open(raw_path) as img:
                tensor = transform(img)
                images.append(tensor)

        images = torch.stack(images)
        os.makedirs(preprocessed_path, exist_ok=True)
        torch.save(images, f"{preprocessed_path}/images.pt")


if __name__ == "__main__":
    raw_data_path = Path("data/raw")
    preprocessed_path = Path("data/processed")
    Pokemon.preprocess(raw_data_path, preprocessed_path)
