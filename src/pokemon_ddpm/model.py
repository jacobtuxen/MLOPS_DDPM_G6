from typing import Tuple, Union

import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from pokemon_ddpm import _PATH_TO_MODELS


def get_models(
    model_name: Union[str, None],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_train_steps: int = 1000,
) -> Tuple[DDPMPipeline, UNet2DModel]:
    """Return the model and the pipeline. If a model name is provided,
    load the model from disk. Otherwise, create a new model."""

    if model_name is None:
        unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3).to(device)
        ddpm = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=num_train_steps))
    else:
        ddpm = torch.load(f"{_PATH_TO_MODELS}/{model_name}", map_location=device)
        unet = ddpm.unet

    return ddpm, unet
