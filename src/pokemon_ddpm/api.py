import io
from typing import Generator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pokemon_ddpm import _PATH_TO_MODELS
from pokemon_ddpm.model import get_models


def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """Load model and classes, and create database file."""
    global ddpm
    ddpm, _ = get_models(model_name=None)
    ddpm.from_pretrained(pretrained_model_name_or_path=_PATH_TO_MODELS, use_safetensors=False)

    yield

    del ddpm


app = FastAPI(lifespan=lifespan)


@app.get("/sample")
def sample():
    print("Sampling from the model...")
    num_samples = 1
    samples = ddpm(batch_size=num_samples)

    pil_image = samples[0][0]

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
