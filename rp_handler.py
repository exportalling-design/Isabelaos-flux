# rp_handler.py – Worker Serverless de IsabelaOS Studio (FLUX + SD Inpaint legacy)

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
# (diffusers todavía lo usa en algunas rutas)
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
    FluxPipeline,
    StableDiffusionInpaintPipeline,
)
import runpod

# ---------------------------------------------------------
# CONFIG GENERAL
# ---------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Volume base (RunPod Network Volume)
# OJO: en serverless RunPod monta /runpod/volumes/<volume_name>
# Si tu volumen se llama "isabela-video", el path será:
VOLUME_NAME = os.getenv("ISE_VOLUME_NAME", "isabela-video")
VOLUME_DIR = Path(f"/runpod/volumes/{VOLUME_NAME}")

# Cache HF (IMPORTANTE para que NO use /workspace)
HF_CACHE_DIR = Path(os.getenv("ISE_HF_CACHE_DIR", str(VOLUME_DIR / "huggingface")))
TORCH_CACHE_DIR = Path(os.getenv("ISE_TORCH_HOME", str(VOLUME_DIR / "torch")))

# Carpeta de outputs temporales (si quieres guardar debug en volumen)
# (Tu backend debería subir a Supabase; esto es opcional)
OUTPUTS_DIR = Path(os.getenv("ISE_OUTPUTS_DIR", str(VOLUME_DIR / "outputs")))

for d in [HF_CACHE_DIR, TORCH_CACHE_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Para FLUX
FLUX_MODEL_ID = os.getenv("ISE_FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")

# Para SD Inpaint (mantener lo que ya te funciona)
INPAINT_MODEL_ID = os.getenv("ISE_INPAINT_MODEL_ID", "runwayml/stable-diffusion-inpainting")

# Pipelines cacheados
pipe_flux: FluxPipeline | None = None
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

def _set_cache_env():
    # Fuerza a HF/diffusers/transformers a usar volumen
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("DIFFUSERS_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))


def get_flux_pipeline() -> FluxPipeline:
    global pipe_flux
    if pipe_flux is not None:
        return pipe_flux

    _set_cache_env()

    # FLUX suele ir mejor en bf16
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    pipe_flux = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE_DIR),   # clave: volumen
    )

    if DEVICE == "cuda":
        pipe_flux = pipe_flux.to("cuda")

    return pipe_flux


def get_inpaint_pipeline() -> StableDiffusionInpaintPipeline:
    global pipe_inpaint
    if pipe_inpaint is not None:
        return pipe_inpaint

    _set_cache_env()

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE_DIR),   # también al volumen
        safety_checker=None,
        feature_extractor=None,
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
# TXT2IMG CON FLUX (DEFAULT)
# ---------------------------------------------------------

def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux_pipeline()

    prompt = (input_data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Falta prompt")

    # FLUX schnell: 1-4 steps recomendado
    steps = int(input_data.get("steps", 4))
    steps = max(1, min(steps, 8))

    # Tamaño: FLUX puede con 1024x1024; usamos defaults igual que tu contrato
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))
    width, height = ensure_size_for_sd(width, height, max_side=1024)

    # guidance_scale en schnell normalmente 0.0
    guidance = float(input_data.get("guidance_scale", 0.0))
    guidance = 0.0 if guidance < 0 else guidance

    # seed opcional
    seed = input_data.get("seed", None)
    generator = None
    if seed is not None:
        try:
            seed_int = int(seed)
            generator = torch.Generator("cpu").manual_seed(seed_int)
        except Exception:
            generator = None

    max_seq = int(input_data.get("max_sequence_length", 256))
    max_seq = max(64, min(max_seq, 512))

    # Autocast (cuda)
    if DEVICE == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=height,
                width=width,
                max_sequence_length=max_seq,
                generator=generator,
            )
    else:
        out = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=height,
            width=width,
            max_sequence_length=max_seq,
            generator=generator,
        )

    img = out.images[0]

    # Guardado opcional en volumen (solo para debug). Tu backend debería subir a Supabase.
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"isabelaos_flux_{ts}.png"
    try:
        img.save(path)
        file_path = str(path)
    except Exception:
        file_path = ""

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": file_path,
        "mode": "txt2img",
        "model": "flux",
        "steps": steps,
        "width": width,
        "height": height,
    }

# ---------------------------------------------------------
# NAVIDAD ESTUDIO (MISMA LÓGICA, INPAINT SD)
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

    if DEVICE == "cuda":
        with torch.autocast("cuda"):
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
    else:
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
    path = OUTPUTS_DIR / f"isabelaos_navidad_estudio_{ts}.png"
    try:
        final_image.save(path)
        file_path = str(path)
    except Exception:
        file_path = ""

    return {
        "image_b64": encode_image_to_b64(final_image),
        "file_path": file_path,
        "mode": "navidad_estudio",
    }

# ---------------------------------------------------------
# HEADSHOT PRO (MISMA LÓGICA, INPAINT SD)
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

    if DEVICE == "cuda":
        with torch.autocast("cuda"):
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
    else:
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
    path = OUTPUTS_DIR / f"isabelaos_headshot_{ts}.png"
    try:
        img.save(path)
        file_path = str(path)
    except Exception:
        file_path = ""

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": file_path,
        "mode": "headshot_pro",
        "style": style,
    }

# ---------------------------------------------------------
# HANDLER PRINCIPAL (MISMO CONTRATO)
# ---------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = input_data.get("action")

        if action == "health":
            return {
                "message": "isabelaOs worker online",
                "device": DEVICE,
                "volume": str(VOLUME_DIR),
                "hf_cache": str(HF_CACHE_DIR),
            }

        if action == "navidad_estudio":
            return handle_navidad_estudio(input_data)

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        # default: txt2img (FLUX)
        return handle_txt2img(input_data)

    except Exception as e:
        print("[rp_handler] ERROR:", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
