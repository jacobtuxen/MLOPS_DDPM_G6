from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from pokemon_ddpm.model import get_models
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from data import PokemonDataset
from torch.utils.data import Dataset, DataLoader

from pokemon_ddpm import __PATH_TO_DATA__, __PATH_TO_ROOT__

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    model=None, #fix this
    lr: float = 1e-3,
    lr_warmup_steps: int = 10,
    batch_size: int = 32,
    epochs: int = 10,
    save_model: bool = False,
    train_set: Dataset = PokemonDataset(__PATH_TO_DATA__),
) -> None:
    """Train a model on pokemon images."""

    model = model.to(DEVICE)
    model.train()
    train_dataloader = DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=(len(train_dataloader) * epochs)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    for epoch in range(epochs):
        model.train()

        for images in tqdm(train_dataloader, desc="Processing batches"):
            images = images.to(DEVICE)
            noise = torch.randn(images.shape, device=DEVICE)

            timesteps = torch.randint(0, 1000, (images.shape[0],))
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps.float(), return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch} Loss: {loss.item()}")  # TODO: Add logging (pref wandb?)

        if save_model:
            torch.save(model.state_dict(), Path(__PATH_TO_ROOT__) / "models" / "model.pt")


if __name__ == "__main__":
    ddpmp, unet = get_models(model_name=None, device=DEVICE)
    train(model=unet, train_set=PokemonDataset(__PATH_TO_DATA__))
