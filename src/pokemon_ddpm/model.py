from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch

class DDPM(torch.nn.Module):
    def __init__(self) -> None:
        super(DDPM, self).__init__()
        self.unet = UNet2DModel(
          sample_size=32,
          in_channels=3,
          out_channels=3,
          layers_per_block=2,
          block_out_channels=(128, 128, 256, 256, 512, 512),
          down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
          ),
          up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
          ),
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.ddpm = DDPMPipeline(
            unet=self.unet,
            scheduler=self.scheduler,
        )

    def forward(self, x):
        return self.ddpm(x)
    