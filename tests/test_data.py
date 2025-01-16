import os

import pytest
from torch.utils.data import Dataset

from pokemon_ddpm import _PATH_TO_DATA
from pokemon_ddpm.data import PokemonDataset


@pytest.mark.skipif(not os.path.exists(_PATH_TO_DATA), reason="Data files not found")
def test_dataset_type():
    """Test the MyDataset class."""
    dataset = PokemonDataset(_PATH_TO_DATA)
    assert isinstance(dataset, Dataset), "The dataset is not a valid PyTorch Dataset"


@pytest.mark.skipif(not os.path.exists(_PATH_TO_DATA), reason="Data files not found")
def test_data():
    dataset = PokemonDataset(_PATH_TO_DATA)
    assert len(dataset) == 819, f"len(dataset_test)={len(dataset)}"
    assert all(sample.shape == (3, 32, 32) for sample in dataset), "Images have incorrect shape"
