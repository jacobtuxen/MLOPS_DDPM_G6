from pokemon_ddpm import _PATH_TO_MODELS
from pokemon_ddpm.model import PokemonDDPM


def sample_model(model, num_samples):
    """Sample from the model."""
    samples = model(batch_size=num_samples)
    return samples


if __name__ == "__main__":
    model = PokemonDDPM()
    model.ddpm.from_pretrained(pretrained_model_name_or_path=_PATH_TO_MODELS, use_safetensors=False)
    num_samples = 1
    samples = model.sample()
    samples[0][0].show()  # awful code, but it works
