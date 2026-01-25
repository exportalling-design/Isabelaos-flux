import os
import io
import base64
from datetime import datetime
from typing import Dict, Any

import torch
from PIL import Image
from diffusers import FluxPipeline
import runpod

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

pipe: FluxPipeline | None = None

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# -------------------------------------------------
# PIPELINE
# -------------------------------------------------
def get_pipe() -> FluxPipeline:
    global pipe
    if pipe is not None:
        return pipe

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    return pipe

# -------------------------------------------------
# HANDLER
# -------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "")

    steps = int(input_data.get("steps", 4))
    seed = int(input_data.get("seed", 0))

    generator = torch.Generator("cpu").manual_seed(seed)

    pipe = get_pipe()

    image = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=steps,
        max_sequence_length=256,
        generator=generator
    ).images[0]

    return {
        "image_b64": encode_image(image),
        "model": "flux-1-schnell",
        "steps": steps
    }

runpod.serverless.start({"handler": handler})
