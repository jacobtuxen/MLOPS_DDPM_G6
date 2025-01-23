import io
from typing import Generator

from diffusers import DDPMPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Summary, make_asgi_app


def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """Load model and classes, and create database file."""
    global model
    model = DDPMPipeline.from_pretrained(pretrained_model_name_or_path="models/", use_safetensors=False)

    yield

    del model

registry = CollectorRegistry()
error_counter = Counter("pokemon_ddpm_errors", "Number of errors that occurred", registry=registry)
summary = Summary("pokemon_ddpm_request_processing_seconds", "Time spent processing request", registry=registry)

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app(registry=registry))


@app.get("/sample")
async def sample():
    try:
        print("Sampling from the model...")
        samples = model(num_inference_steps=200)

        pil_image = samples[0][0]

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))



