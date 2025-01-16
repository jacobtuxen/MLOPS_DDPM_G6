import glob
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PokemonDataset(Dataset):
    """Pokémon sprite dataset."""

    def __init__(self, data_path: Path) -> None:
        self.image_paths = list(glob.glob(os.path.join(data_path, "*.jpg")))
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.images = []
        self._preprocess()

    def _preprocess(self) -> None:
        for image_path in self.image_paths:
            image = Image.open(image_path)
            image = self.transform(image)
            self.images.append(image)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index]
