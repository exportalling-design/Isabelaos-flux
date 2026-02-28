# rp_handler.py – IsabelaOS Studio (FLUX + SDXL IMG2IMG in ONE Serverless Worker)

import os
import io
import base64
from typing import Dict, Any, Optional

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
from diffusers import FluxPipeline, AutoPipelineForImage2Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# TXT2IMG
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# IMG2IMG (para “misma foto pero mejorada”)
# SDXL es fuerte para producto/estudio manteniendo forma si strength es bajo.
SDXL_IMG2IMG_ID = os.environ.get("SDXL_IMG2IMG_ID", "stabilityai/stable-diffusion-xl-base-1.0")

flux_pipe: Optional[FluxPipeline] = None
img2img_pipe = None

# ---------------------------------------------------------
# PIPELINES (SE CARGAN UNA SOLA VEZ)
# ---------------------------------------------------------
def get_flux() -> FluxPipeline:
    global flux_pipe
    if flux_pipe is not None:
        return flux_pipe

    print("[IsabelaOS] Loading FLUX pipeline...")
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    if DEVICE == "cuda":
        flux_pipe = flux_pipe.to("cuda")
    return flux_pipe


def get_img2img():
    global img2img_pipe
    if img2img_pipe is not None:
        return img2img_pipe

    print("[IsabelaOS] Loading SDXL IMG2IMG pipeline...")
    img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
        SDXL_IMG2IMG_ID,
        torch_dtype=DTYPE,
        cache_dir=os.environ["HF_HUB_CACHE"],
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True,
    )
    if DEVICE == "cuda":
        img2img_pipe = img2img_pipe.to("cuda")

        # Opcional (si te falta VRAM). Comenta si no lo necesitas:
        # img2img_pipe.enable_xformers_memory_efficient_attention()
        # img2img_pipe.enable_model_cpu_offload()

    return img2img_pipe


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def clamp_size(img: Image.Image, max_side: int = 1024) -> Image.Image:
    # SDXL va mejor con múltiplos de 8 y tamaños ~1024
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw = int((w * scale) // 8 * 8)
    nh = int((h * scale) // 8 * 8)
    if nw < 256: nw = 256
    if nh < 256: nh = 256
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


# ---------------------------------------------------------
# TXT2IMG (FLUX)
# ---------------------------------------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux()

    prompt = input_data.get("prompt", "")
    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast("cuda"):
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                ).images[0]
        else:
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
# IMG2IMG (SDXL) – “MISMA FOTO” PERO MEJORADA A ESTUDIO
# ---------------------------------------------------------
def handle_headshot_pro(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_img2img()

    if not input_data.get("image_b64"):
        return {"error": "MISSING_IMAGE_B64"}

    init_img = decode_image(input_data["image_b64"])
    init_img = clamp_size(init_img, max_side=int(input_data.get("max_side", 1024)))

    # PROMPT “foto de producto en estudio”
    # Lo puedes personalizar por categoría: comida, vidrio, herramientas, etc.
    user_style = (input_data.get("style") or "corporate").strip().lower()

    # Prompt base (mantiene foto, mejora luz/estudio)
    if user_style == "creative":
        prompt = (
            "commercial product photography, studio lighting, softbox light, "
            "high detail, clean composition, natural colors, sharp focus, "
            "premium advertising photo, seamless background, realistic"
        )
    elif user_style == "influencer":
        prompt = (
            "product photo, natural lifestyle studio lighting, clean background, "
            "sharp focus, realistic, premium social media product shot"
        )
    else:
        prompt = (
            "commercial product photography, professional studio lighting, "
            "soft shadows, clean seamless background, sharp focus, realistic, "
            "high detail, premium e-commerce photo"
        )

    # Negativos para evitar que invente paisajes, personas, etc.
    negative = (
        "waterfall, canyon, landscape, people, face, hands, text, logo, watermark, "
        "extra objects, clutter, messy background, low quality, blurry, cartoon, anime"
    )

    # Parámetros CLAVE para “MISMA FOTO”:
    # - strength bajo (0.20–0.40) mantiene composición/objeto
    # - guidance moderado
    steps = int(input_data.get("steps", 25))
    guidance = float(input_data.get("guidance", 5.5))
    strength = float(input_data.get("strength", 0.28))  # 🔥 empieza aquí para mantener “misma foto”
    seed = input_data.get("seed", None)

    generator = None
    if seed is not None:
        try:
            seed = int(seed)
            generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)
        except:
            generator = None

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast("cuda"):
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=init_img,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                ).images[0]
        else:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

    return {
        "image_b64": encode_image(out),
        "mode": "img2img_sdxl_product_studio",
        "engine": "sdxl_img2img",
        "params": {
            "steps": steps,
            "guidance": guidance,
            "strength": strength,
            "seed": seed,
            "style": user_style,
            "size": list(init_img.size),
        },
    }


# ---------------------------------------------------------
# HANDLER PRINCIPAL
# ---------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = (input_data.get("action") or "").strip()

        if action == "health":
            return {"message": "IsabelaOS worker online (FLUX txt2img + SDXL img2img)"}

        # ✅ Tu backend manda action: "headshot_pro"
        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        # Default: txt2img (lo que ya tenías)
        return handle_txt2img(input_data)

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
