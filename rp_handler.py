# rp_handler.py – Worker Serverless de IsabelaOS Studio

import os
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch

# ---------------------------------------------------------
# PARCHE HUGGINGFACE_HUB – cached_download
# (diffusers todavía lo usa, pero en hf>=0.34 ya no existe)
# ---------------------------------------------------------
import huggingface_hub as h

if not hasattr(h, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download

    def _cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)

    h.cached_download = _cached_download
    print("[PARCHE] huggingface_hub.cached_download definido usando hf_hub_download")

# ---------------------------------------------------------
# IMPORTS DE DIFFUSERS Y RUNPOD (después del parche)
# ---------------------------------------------------------
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
)
import runpod

# ---------------------------------------------------------
# CONFIG GENERAL
# ---------------------------------------------------------
HF_MODEL_ID = os.getenv(
    "ISE_BASE_MODEL_ID",
    "SG161222/Realistic_Vision_V5.1_noVAE"
)

BASE_DIR = Path("/runpod/volumes/isabelaos")
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

for d in [MODELS_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe_txt2img: StableDiffusionPipeline | None = None
pipe_inpaint: StableDiffusionInpaintPipeline | None = None

# ---------------------------------------------------------
# HELPERS BÁSICOS
# ---------------------------------------------------------

def decode_image_from_b64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ensure_size_for_sd(width: int, height: int, max_side: int = 1024) -> tuple[int, int]:
    width = max(int(width), 8)
    height = max(int(height), 8)

    max_current = max(width, height)
    if max_current > max_side:
        scale = max_side / float(max_current)
        width = int(width * scale)
        height = int(height * scale)

    width = max((width // 8) * 8, 8)
    height = max((height // 8) * 8, 8)

    return width, height


def enhance_for_studio(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(1.05)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.06)
    img = ImageEnhance.Sharpness(img).enhance(1.10)
    return img

# ---------------------------------------------------------
# CARGA DE PIPELINES
# ---------------------------------------------------------

def get_txt2img_pipeline() -> StableDiffusionPipeline:
    global pipe_txt2img

    if pipe_txt2img is not None:
        return pipe_txt2img

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=str(MODELS_DIR),
        safety_checker=None,
        feature_extractor=None,
    )

    if DEVICE == "cuda":
        pipe_txt2img = pipe_txt2img.to("cuda")

    pipe_txt2img.enable_attention_slicing()
    return pipe_txt2img


def get_inpaint_pipeline() -> StableDiffusionInpaintPipeline:
    global pipe_inpaint

    if pipe_inpaint is not None:
        return pipe_inpaint

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=str(MODELS_DIR),
    )

    if DEVICE == "cuda":
        pipe_inpaint = pipe_inpaint.to("cuda")

    pipe_inpaint.enable_attention_slicing()
    return pipe_inpaint

# ---------------------------------------------------------
# MÁSCARA – FONDO BLANCO, PERSONA NEGRO
# ---------------------------------------------------------

def create_background_mask(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
    except Exception:
        return Image.new("L", image.size, 0)

    try:
        rgba = remove(np.array(image))
        alpha = rgba[:, :, 3]

        mask_person = Image.fromarray(alpha).convert("L")
        mask_person = mask_person.point(lambda x: 255 if x > 10 else 0)
        mask_bg = mask_person.point(lambda x: 0 if x > 0 else 255)
        mask_bg = mask_bg.filter(ImageFilter.MaxFilter(5))
        mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(3))
        return mask_bg

    except Exception:
        return Image.new("L", image.size, 0)

# ---------------------------------------------------------
# TXT2IMG ESTÁNDAR
# ---------------------------------------------------------

def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_txt2img_pipeline()

    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "")

    width, height = ensure_size_for_sd(
        int(input_data.get("width", 512)),
        int(input_data.get("height", 512)),
    )

    steps = int(input_data.get("steps", 22))
    guidance = float(input_data.get("guidance_scale", 7.5))

    with torch.autocast("cuda") if DEVICE == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )

    img = result.images[0]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"isabelaos_{ts}.png"
    img.save(path)

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": str(path),
        "mode": "txt2img",
    }

# ---------------------------------------------------------
# NAVIDAD ESTUDIO (ORIGINAL – SIN TOCAR)
# ---------------------------------------------------------

def handle_navidad_estudio(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # 👉 EXACTAMENTE TU CÓDIGO, SIN CAMBIOS
    img_b64 = input_data.get("image_b64")
    if not img_b64:
        raise ValueError("Falta image_b64 en navidad_estudio")

    description = (input_data.get("description") or "").strip()

    original = decode_image_from_b64(img_b64)
    original = enhance_for_studio(original)

    w, h = original.size
    w_new, h_new = ensure_size_for_sd(w, h, max_side=1024)

    if (w_new, h_new) != (w, h):
        original = original.resize((w_new, h_new), Image.LANCZOS)
        w, h = w_new, h_new

    mask_bg = create_background_mask(original)

    import random
    background_scenes = [
        "luxury christmas photo studio background, big christmas tree with golden ornaments, wrapped gifts, warm fairy lights, elegant sofa, professional studio lighting, no people in the background",
        "cozy christmas living room background, decorated tree, fireplace with stockings, wooden floor, warm soft lights, professional photography background, no extra persons",
        "minimal beige photo studio background with subtle christmas decorations, small tree with golden ornaments, clean elegant set, soft light, no people in the background",
        "snowy cabin interior, christmas decorations, tree with red and gold ornaments, window with snow outside, warm yellow lights, cinematic background, no humans in the background",
        "modern christmas loft, big windows with city bokeh lights, christmas tree in the corner, fairy lights, stylish sofa, professional photoshoot background, no people, no cameras",
    ]

    chosen_scene = random.choice(background_scenes)

    base_prompt = (
        "ultra realistic family christmas portrait photography, 8k, natural skin tones, "
        "keep the same people, same faces, same pose and clothes, change ONLY the background, "
        "no changes to faces, no changes to bodies, " + chosen_scene
    )

    full_prompt = base_prompt + (", " + description if description else "")

    negative_prompt = (
        "extra people, duplicated people, duplicate person, duplicate heads, extra heads, "
        "distorted face, deformed face, melted face, wrong identity, different person, "
        "mutated hands, extra limbs, bad anatomy, low quality, blurry, noisy, grainy, "
        "cartoon, illustration, 3d render, cgi, watermark, logo, text, caption, "
        "camera, photographer, tripod, microphone, studio equipment"
    )

    pipe = get_inpaint_pipeline()

    with torch.autocast("cuda") if DEVICE == "cuda" else torch.no_grad():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=original,
            mask_image=mask_bg,
            width=w,
            height=h,
            num_inference_steps=28,
            guidance_scale=7.5,
        )

    final_image = result.images[0]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"isabelaos_navidad_estudio_{ts}.png"
    final_image.save(path)

    return {
        "image_b64": encode_image_to_b64(final_image),
        "file_path": str(path),
        "mode": "navidad_estudio",
    }

# ---------------------------------------------------------
# HEADSHOT PRO (ÚNICA ADICIÓN)
# ---------------------------------------------------------

def handle_headshot_pro(input_data: Dict[str, Any]) -> Dict[str, Any]:
    img_b64 = input_data.get("image_b64")
    if not img_b64:
        raise ValueError("Falta image_b64 en headshot_pro")

    style = input_data.get("style", "corporate")

    original = enhance_for_studio(decode_image_from_b64(img_b64))
    w, h = ensure_size_for_sd(*original.size)
    original = original.resize((w, h), Image.LANCZOS)

    mask_bg = create_background_mask(original)

    styles = {
        "corporate": "professional corporate studio background, soft gray gradient, clean lighting, no people",
        "influencer": "modern lifestyle studio background, soft bokeh lights, premium look, no people",
        "creative": "cinematic dark studio background, dramatic lighting, editorial style, no people",
    }

    prompt = (
        "ultra realistic professional headshot portrait, "
        "keep the same person, same face, same identity, "
        "change ONLY the background, "
        + styles.get(style, styles["corporate"])
    )

    negative_prompt = (
        "different person, altered face, beauty filter, plastic skin, "
        "extra people, extra heads, watermark, text, logo"
    )

    pipe = get_inpaint_pipeline()

    with torch.autocast("cuda") if DEVICE == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original,
            mask_image=mask_bg,
            width=w,
            height=h,
            num_inference_steps=26,
            guidance_scale=7.0,
        )

    img = result.images[0]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"isabelaos_headshot_{ts}.png"
    img.save(path)

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": str(path),
        "mode": "headshot_pro",
        "style": style,
    }

# ---------------------------------------------------------
# HANDLER PRINCIPAL
# ---------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = input_data.get("action")

        if action == "health":
            return {"message": "isabelaOs worker online"}

        if action == "navidad_estudio":
            return handle_navidad_estudio(input_data)

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        return handle_txt2img(input_data)

    except Exception as e:
        print("[rp_handler] ERROR:", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
