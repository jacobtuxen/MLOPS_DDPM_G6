import torch
from src.pokemon_ddpm.model import get_models
import pytest
from diffusers import DDPMPipeline, UNet2DModel, ImagePipelineOutput
from PIL import Image

def test_get_models():
    device = torch.device("cpu")
    ddpm, unet = get_models(model_name=None, device=device)

    # Ensure the models are correctly instantiated
    assert isinstance(ddpm, DDPMPipeline), "DDPM model is not an instance of DDPMPipeline"
    assert isinstance(unet, UNet2DModel), "UNet model is not an instance of UNet2DModel"

    # Perform a forward pass with the DDPM pipeline
    with torch.no_grad():
        # Generate an output from the pipeline
        output = ddpm(num_inference_steps=1)

    # Ensure the output is an instance of ImagePipelineOutput
    assert isinstance(output, ImagePipelineOutput), f"Expected output to be an instance of ImagePipelineOutput, but got {type(output)}"

    # Access the generated images
    generated_images = output.images

    # Ensure the output images is a list and contains at least one image
    assert isinstance(generated_images, list), f"Expected 'images' to be a list, but got {type(generated_images)}"
    assert len(generated_images) == 1, f"Expected 1 image, but got {len(generated_images)}"

    # Get the first image and assert it's a PIL.Image object
    generated_image = generated_images[0]
    assert isinstance(generated_image, Image.Image), f"Expected generated image to be a PIL.Image, but got {type(generated_image)}"

    # Optionally, check the size of the image (if necessary)
    expected_size = (32, 32)  # 32x32 image size
    assert generated_image.size == expected_size, (
        f"Expected image size {expected_size}, but got {generated_image.size}"
    )






