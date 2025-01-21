import io
from typing import Generator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pokemon_ddpm import _PATH_TO_MODELS
from pokemon_ddpm.model import PokemonDDPM


def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """Load model and classes, and create database file."""
    global model
    model = PokemonDDPM()
    model.ddpm.from_pretrained(pretrained_model_name_or_path='models/', use_safetensors=False)

    yield

    del model


app = FastAPI(lifespan=lifespan)


@app.get("/sample")
def sample():
    print("Sampling from the model...")
    samples = model.sample()

    pil_image = samples[0][0]

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
