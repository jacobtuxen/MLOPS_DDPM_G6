import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import typer
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from model import DDPM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    lr: float = 1e-3, lr_warmup_steps: int = 10, batch_size: int = 32, epochs: int = 10
) -> None:
    """Train a model on pokemon images."""
    print(f"{lr=}, {batch_size=}, {epochs=}")
    
    ddpm = DDPM()
    model = ddpm.unet.to(DEVICE)
    train_set = torch.load("data/processed/images.pt")

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=(len(train_dataloader) * epochs)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    global_step = 0

    statistics = {"train_loss": []}

    for epoch in range(epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(DEVICE)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residuals
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            statistics["train_loss"].append(loss.item())

            if step % 100 == 0:
                print(f"Epoch {epoch}, iter {step}, loss: {loss.item()}")

            plt.plot(statistics["train_loss"])
            plt.title("Train loss")
            plt.savefig("reports/figures/train_loss.png")
    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train()
