import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from pytorch_lightning import LightningModule


class PokemonDDPM(LightningModule):
    def __init__(self, timesteps: int = 200):
        super().__init__()
        self.timesteps = timesteps
        self.unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3)
        self.ddpm = DDPMPipeline(unet=self.unet, scheduler=DDPMScheduler(num_train_timesteps=self.timesteps))
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.timesteps)
        self.criterium = torch.nn.MSELoss()

    def forward(self, x, t):
        return self.unet(x, t, return_dict=False)[0]

    def parameters(self):
        return self.unet.parameters()

    def sample(self):
        return self.ddpm(batch_size=1, num_inference_steps=self.timesteps)

    def training_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        images = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, self.timesteps, (images.shape[0],), device=noise.device)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps.float())
        loss = self.criterium(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def save_model(self, path) -> None:
        self.ddpm.save_pretrained(save_directory=path, safe_serialization=False)
