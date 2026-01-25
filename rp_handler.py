# rp_handler.py – Worker Serverless (FLUX txt2img) para IsabelaOS Studio

import os
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch

# ---------------------------------------------------------
# PARCHE HUGGINGFACE_HUB – cached_download (compat)
# ---------------------------------------------------------
import huggingface_hub as h

if not hasattr(h, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download

    def _cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)

    h.cached_download = _cached_download
    print("[PATCH] huggingface_hub.cached_download definido usando hf_hub_download")

# ---------------------------------------------------------
# DIFFUSERS / RUNPOD
# ---------------------------------------------------------
from diffusers import FluxPipeline, StableDiffusionInpaintPipeline
import runpod

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Donde cachea modelos e imágenes:
# - sin volumen: queda dentro del contenedor (rápido con Flashboot, pero no persistente)
# - con volumen: setea ISE_BASE_DIR="/runpod/volumes/isabelaos"
BASE_DIR = Path(os.getenv("ISE_BASE_DIR", "/workspace/isabelaos_cache"))
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
for d in [MODELS_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ✅ Modelo FLUX
FLUX_MODEL_ID = os.getenv("ISE_FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")

# Inpaint (lo dejo como antes, independiente de FLUX)
INPAINT_MODEL_ID = os.getenv("ISE_INPAINT_MODEL_ID", "runwayml/stable-diffusion-inpainting")

# dtype recomendado
# En GPUs grandes puedes usar bfloat16 sin problema. Si algo falla, cambia a float16.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

pipe_flux: Optional[FluxPipeline] = None
pipe_inpaint: Optional[StableDiffusionInpaintPipeline] = None

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def decode_image_from_b64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def encode_image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ensure_size_for_flux(width: int, height: int, max_side: int = 1024) -> tuple[int, int]:
    # FLUX suele ir mejor en múltiplos de 16
    width = max(int(width), 64)
    height = max(int(height), 64)

    max_current = max(width, height)
    if max_current > max_side:
        scale = max_side / float(max_current)
        width = int(width * scale)
        height = int(height * scale)

    width = max((width // 16) * 16, 64)
    height = max((height // 16) * 16, 64)
    return width, height

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
# PIPELINES
# ---------------------------------------------------------
def get_flux_pipeline() -> FluxPipeline:
    global pipe_flux
    if pipe_flux is not None:
        return pipe_flux

    print(f"[FLUX] Cargando modelo: {FLUX_MODEL_ID}")
    pipe_flux = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE,
        cache_dir=str(MODELS_DIR),
    )

    if DEVICE == "cuda":
        pipe_flux = pipe_flux.to("cuda")

    # Opcional: reduce picos de VRAM
    pipe_flux.enable_attention_slicing()
    return pipe_flux

def get_inpaint_pipeline() -> StableDiffusionInpaintPipeline:
    global pipe_inpaint
    if pipe_inpaint is not None:
        return pipe_inpaint

    print(f"[INPAINT] Cargando: {INPAINT_MODEL_ID}")
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=str(MODELS_DIR),
        safety_checker=None,
        feature_extractor=None,
    )

    if DEVICE == "cuda":
        pipe_inpaint = pipe_inpaint.to("cuda")

    pipe_inpaint.enable_attention_slicing()
    return pipe_inpaint

# ---------------------------------------------------------
# MÁSCARA (igual que tu lógica)
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
# TXT2IMG con FLUX
# ---------------------------------------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux_pipeline()

    prompt = (input_data.get("prompt") or "").strip()
    negative_prompt = (input_data.get("negative_prompt") or "").strip()

    width, height = ensure_size_for_flux(
        int(input_data.get("width", 1024)),
        int(input_data.get("height", 1024)),
        max_side=int(input_data.get("max_side", 1024)),
    )

    # ⚙️ Recomendación:
    # - FLUX.1-schnell: steps 4–8 suele ser suficiente (rápido y nítido)
    # - FLUX.1-dev: steps 25–35 (más lento, más calidad)
    steps = int(input_data.get("steps", 8))

    # FLUX maneja guidance distinto; valores bajos suelen funcionar bien
    guidance = float(input_data.get("guidance_scale", 3.5))

    seed = input_data.get("seed", None)
    generator = None
    if seed is not None:
        try:
            seed_int = int(seed)
            generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed_int)
        except Exception:
            generator = None

    # Ejecutar
    if DEVICE == "cuda":
        with torch.autocast("cuda", dtype=DTYPE):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
    else:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

    img = result.images[0]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"isabelaos_flux_{ts}.png"
    img.save(path)

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": str(path),
        "mode": "txt2img",
        "model": "flux",
        "model_id": FLUX_MODEL_ID,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
    }

# ---------------------------------------------------------
# NAVIDAD ESTUDIO (tu lógica, solo ajusto llamadas a inpaint)
# ---------------------------------------------------------
def handle_navidad_estudio(input_data: Dict[str, Any]) -> Dict[str, Any]:
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
            num_inference_steps=int(input_data.get("steps", 28)),
            guidance_scale=float(input_data.get("guidance_scale", 7.5)),
        )

    final_image = result.images[0]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"isabelaos_navidad_estudio_{ts}.png"
    final_image.save(path)

    return {
        "image_b64": encode_image_to_b64(final_image),
        "file_path": str(path),
        "mode": "navidad_estudio",
        "model": "sd_inpaint",
        "model_id": INPAINT_MODEL_ID,
    }

# ---------------------------------------------------------
# HEADSHOT PRO (igual idea)
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
            num_inference_steps=int(input_data.get("steps", 26)),
            guidance_scale=float(input_data.get("guidance_scale", 7.0)),
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
        "model": "sd_inpaint",
        "model_id": INPAINT_MODEL_ID,
    }

# ---------------------------------------------------------
# HANDLER
# ---------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = input_data.get("action")

        if action == "health":
            return {"message": "isabelaos-flux worker online", "device": DEVICE, "model_id": FLUX_MODEL_ID}

        if action == "navidad_estudio":
            return handle_navidad_estudio(input_data)

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        # default: txt2img con FLUX
        return handle_txt2img(input_data)

    except Exception as e:
        print("[rp_handler] ERROR:", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})