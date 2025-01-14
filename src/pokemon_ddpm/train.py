from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from pokemon_ddpm import __PATH_TO_DATA__, __PATH_TO_ROOT__

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    model=None,
    lr: float = 1e-3,
    lr_warmup_steps: int = 10,
    batch_size: int = 32,
    epochs: int = 10,
    save_model: bool = False,
) -> None:
    """Train a model on pokemon images."""

    model = model.to(DEVICE)
    model.train()

    train_set = torch.load(__PATH_TO_DATA__ / "processed" / "images.pt")  # FIX THIS
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

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
    unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3).to(DEVICE)
    DDPM = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=1000))
    train(model=unet)
