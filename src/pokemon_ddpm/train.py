from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

from pokemon_ddpm import _PATH_TO_DATA, _PATH_TO_MODELS
from pokemon_ddpm.data import PokemonDataset
from pokemon_ddpm.model import get_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
wandb.login()

def train(
    model=None,  # fix this
    lr: float = 1e-3,
    lr_warmup_steps: int = 10,
    batch_size: int = 32,
    epochs: int = 10,
    save_model: bool = False,
    train_set: Dataset = PokemonDataset(_PATH_TO_DATA),
    wandb_active: bool = False,
) -> None:
    """Train a model on pokemon images."""

    run = wandb.init(
        project="pokemon-ddpm",
        config={
            "lr": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "batch_size": batch_size,
            "epochs": epochs,
            "save_model": save_model,
        },
    )

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
        epoch_loss = 0

        for images in tqdm(train_dataloader, desc="Processing batches"):
            images = images.to(DEVICE)
            noise = torch.randn(images.shape, device=DEVICE)

            timesteps = torch.randint(0, 1000, (images.shape[0],), device=DEVICE)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps.float(), return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            
            if wandb_active:
                wandb.log({
                    "train/train_loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    })

        print(f"Epoch {epoch} Loss: {loss.item()}")  # TODO: Add logging (pref wandb?)

        if save_model:
            torch.save(model.state_dict(), Path(_PATH_TO_MODELS) / "models" / "model.pt")
            if wandb_active:
                wandb.save(str(Path(_PATH_TO_MODELS) / "models" / "model.pt"))
        
    wandb.finish()

def sweep_train(config=None):
    """Function to be executed during a wandb sweep."""
    with wandb.init(config=config):
        config = wandb.config
        ddpmp, unet = get_models(model_name=None, device=DEVICE)
        train(
            model=unet,
            lr=config.lr,
            lr_warmup_steps=config.lr_warmup_steps,
            batch_size=config.batch_size,
            epochs=config.epochs,
            save_model=config.save_model,
            train_set=PokemonDataset(_PATH_TO_DATA),
            wandb_active=True,
        )



if __name__ == "__main__":
    ddpmp, unet = get_models(model_name=None, device=DEVICE)
    train(model=unet, train_set=PokemonDataset(_PATH_TO_DATA))
    sweep_id = wandb.sweep(sweep_config="configs/sweep.yaml", project="pokemon-ddpm")
    wandb.agent(sweep_id, function=sweep_train)
