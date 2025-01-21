import torch
from torch.utils.data import Dataset

from pokemon_ddpm.model import PokemonDDPM
from pokemon_ddpm.train import train


class DummyDataset(Dataset):
    """A dummy dataset with a single data point."""

    def __init__(self):
        super().__init__()
        self.data = [torch.zeros((3, 32, 32))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def test_train():
    dummy_dataset = DummyDataset()
    model = PokemonDDPM()
    train(model=model, train_set=dummy_dataset, epochs=1)
    assert True
