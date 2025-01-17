import matplotlib.pyplot as plt

from pokemon_ddpm.model import get_models


def sample_model(model, num_samples):
    """Sample from the model."""
    samples = model(batch_size=num_samples)
    return samples


if __name__ == "__main__":
    ddpm, unet = get_models(model_name=None)
    num_samples = 1
    samples = sample_model(ddpm, num_samples=num_samples)
    samples[0][0].show()  # awful code, but it works
