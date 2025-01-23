import torch

from pokemon_ddpm import _PATH_TO_MODELS, _PATH_TO_OUTPUT
from pokemon_ddpm.model import PokemonDDPM


def sample_model(model: any, num_samples: int) -> any:
    """Sample from the model."""
    samples = model(batch_size=num_samples)
    return samples


if __name__ == "__main__":
    model = PokemonDDPM()
    model.from_pretrained(pretrained_model_name_or_path=_PATH_TO_MODELS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.ddpm.to(device)
    samples = model.sample()
    samples[0][0].save(_PATH_TO_OUTPUT / "eval.py")  # awful code, but it works
