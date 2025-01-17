from pathlib import Path
import os

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pokemon_ddpm import _PATH_TO_DATA, _PATH_TO_MODELS
from pokemon_ddpm.data import PokemonDataset
from pokemon_ddpm.model import get_models
from pokemon_ddpm.utils import setup_wandb_sweep, log_training

def train(
    model=None,
    lr=1e-4,
    lr_warmup_steps=1000,
    batch_size=32,
    epochs=10,
    save_model=False,
    train_set: Dataset = PokemonDataset(_PATH_TO_DATA),
    wandb_active: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Train a model on pokemon images."""

    model = model.to(device)
    model.train()
    train_dataloader = DataLoader(train_set, batch_size=batch_size)        

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=(len(train_dataloader) * epochs)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images in tqdm(train_dataloader, desc="Processing batches"):
            images = images.to(device)
            noise = torch.randn(images.shape, device=device)

            timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps.float(), return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            
        log_training(epoch, epoch_loss, wandb_active)

        if save_model:
            torch.save(model.state_dict(), Path(_PATH_TO_MODELS) / "models" / "model.pt")      


if __name__ == "__main__":
    ddpmp, unet = get_models(model_name=None)
    wandb_active = True #HYDRAFY
    if wandb_active:
        setup_wandb_sweep(train)
    else:
        train(model=unet, train_set=PokemonDataset(_PATH_TO_DATA))
    
    
