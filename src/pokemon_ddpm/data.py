import glob
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class Pokemon(Dataset):
    """Pokémon sprite dataset."""

    def __init__(self, data_path: Path, transform: Any = None) -> None:
        self.image_paths = list(glob.glob(os.path.join(data_path, '*.jpg')))
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image



# if __name__ == "__main__":

#     transform = transforms.Compose(
#         [
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ]
#     )

#     images = Pokemon("data/pokemon_jpg", transform=transform)
#     train_dataloader = DataLoader(images, batch_size=32, shuffle=True)

#     image_features = next(iter(train_dataloader))
#     img = image_features[0]
#     print(img.shape)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()






# if __name__ == "__main__":
#     raw_data_path = Path("data/raw")
#     preprocessed_path = Path("data/processed")
#     Pokemon = Pokemon(raw_data_path, transform=None)
#     Pokemon.preprocess(raw_data_path, preprocessed_path)
