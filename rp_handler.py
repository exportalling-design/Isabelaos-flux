# rp_handler.py – IsabelaOS Studio (FLUX Serverless Worker)

import os
import io
import base64
from typing import Dict, Any

import torch
from PIL import Image
import runpod

# ---------------------------------------------------------
# FORZAR CACHE A NETWORK VOLUME (CRÍTICO)
# ---------------------------------------------------------
BASE_VOLUME = "/runpod/volumes/isabelaos"

os.environ["HF_HOME"] = f"{BASE_VOLUME}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{BASE_VOLUME}/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{BASE_VOLUME}/huggingface/transformers"
os.environ["DIFFUSERS_CACHE"] = f"{BASE_VOLUME}/huggingface/diffusers"
os.environ["TORCH_HOME"] = f"{BASE_VOLUME}/torch"

for p in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["DIFFUSERS_CACHE"],
    os.environ["TORCH_HOME"],
]:
    os.makedirs(p, exist_ok=True)

# ---------------------------------------------------------
# IMPORTS DESPUÉS DEL CACHE
# ---------------------------------------------------------
from diffusers import FluxPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

pipe: FluxPipeline | None = None

# ---------------------------------------------------------
# PIPELINE (SE CARGA UNA SOLA VEZ)
# ---------------------------------------------------------
def get_pipe() -> FluxPipeline:
    global pipe
    if pipe is not None:
        return pipe

    print("[IsabelaOS] Loading FLUX pipeline...")

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    return pipe

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------------------------------------------------
# TXT2IMG (FLUX)
# ---------------------------------------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_pipe()

    prompt = input_data.get("prompt", "")
    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    with torch.inference_mode(), torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
        ).images[0]

    return {
        "image_b64": encode_image(image),
        "mode": "txt2img_flux",
        "engine": "flux",
    }

# ---------------------------------------------------------
# HANDLER PRINCIPAL
# ---------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = input_data.get("action")

        if action == "health":
            return {"message": "IsabelaOS FLUX worker online"}

        return handle_txt2img(input_data)

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
