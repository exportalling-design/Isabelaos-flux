# rp_handler.py – Worker Serverless de IsabelaOS Studio (REALISTIC + FLUX)
# Mantiene EXACTAMENTE el mismo contrato de entrada/salida:
# - input.action: health | navidad_estudio | headshot_pro | (default txt2img)
# - output: { image_b64, file_path, mode, ... }
#
# ✅ Nuevo: txt2img puede usar FLUX (diffusers FluxPipeline) sin romper tu frontend.
# ✅ Cache/modelos se guardan en el VOLUMEN (para no descargar cada vez).
# ✅ Outputs: por defecto NO se guardan en el volumen. Solo base64. (Puedes activar guardado con env.)

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
# CONFIG CACHE EN VOLUMEN (MUY IMPORTANTE PARA "NO SPACE LEFT")
# ---------------------------------------------------------
# Ajusta el mount de tu Network Volume aquí:
# - Si tu volumen está montado como /runpod/volumes/isabelaos -> deja default
# - Si tu volumen real es /workspace -> pon ISE_VOLUME_MOUNT=/workspace en env vars del endpoint
VOLUME_MOUNT = os.getenv("ISE_VOLUME_MOUNT", "/runpod/volumes/isabelaos")
BASE_DIR = Path(VOLUME_MOUNT)

# Cache dirs dentro del volumen (para que HF/diffusers no usen el disco efímero)
HF_HOME_DIR = BASE_DIR / "hf_home"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
DIFFUSERS_CACHE_DIR = BASE_DIR / "diffusers_cache"
TORCH_HOME_DIR = BASE_DIR / "torch_home"
TMP_DIR = BASE_DIR / "tmp"  # temporal en volumen (opcional)

for d in [HF_HOME_DIR, HF_CACHE_DIR, DIFFUSERS_CACHE_DIR, TORCH_HOME_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("DIFFUSERS_CACHE", str(DIFFUSERS_CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME_DIR))

# ---------------------------------------------------------
# IMPORTS DE DIFFUSERS Y RUNPOD (después del parche)
# ---------------------------------------------------------
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
)
# FluxPipeline puede no existir en versiones viejas -> lo importamos seguro
try:
    from diffusers import FluxPipeline  # type: ignore
except Exception:
    FluxPipeline = None  # type: ignore

import runpod

# ---------------------------------------------------------
# CONFIG GENERAL
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Elegís modelo txt2img por provider:
# - ISE_MODEL_PROVIDER=sd  -> usa SD (Realistic Vision)
# - ISE_MODEL_PROVIDER=flux -> usa FLUX.1-schnell
MODEL_PROVIDER = (os.getenv("ISE_MODEL_PROVIDER", "sd") or "sd").strip().lower()

# SD base (tu actual)
HF_MODEL_ID_SD = os.getenv("ISE_BASE_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE")

# FLUX base (requiere acceso aceptado en HF + token válido)
HF_MODEL_ID_FLUX = os.getenv("ISE_FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")

# Donde se cachean modelos (en volumen)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ Outputs:
# Por defecto NO se guardan. Si querés guardar (solo para debug), pon:
# ISE_SAVE_OUTPUTS=1
SAVE_OUTPUTS = os.getenv("ISE_SAVE_OUTPUTS", "0").strip() == "1"
OUTPUT_DIR = BASE_DIR / "images"
if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PIPELINES CACHE (en memoria del worker)
# ---------------------------------------------------------
pipe_txt2img_sd: Optional[StableDiffusionPipeline] = None
pipe_inpaint: Optional[StableDiffusionInpaintPipeline] = None
pipe_txt2img_flux: Optional[Any] = None  # FluxPipeline

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


def _maybe_save_output(img: Image.Image, prefix: str) -> str:
    """
    Por defecto no guardamos nada (para no mezclar outputs de usuarios).
    Si ISE_SAVE_OUTPUTS=1, se guarda SOLO para debug.
    """
    if not SAVE_OUTPUTS:
        return ""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{prefix}_{ts}.png"
    img.save(path)
    return str(path)

# ---------------------------------------------------------
# CARGA DE PIPELINES
# ---------------------------------------------------------
def get_txt2img_pipeline_sd() -> StableDiffusionPipeline:
    global pipe_txt2img_sd

    if pipe_txt2img_sd is not None:
        return pipe_txt2img_sd

    pipe_txt2img_sd = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID_SD,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=str(MODELS_DIR),
        safety_checker=None,
        feature_extractor=None,
    )

    if DEVICE == "cuda":
        pipe_txt2img_sd = pipe_txt2img_sd.to("cuda")

    pipe_txt2img_sd.enable_attention_slicing()
    return pipe_txt2img_sd


def get_txt2img_pipeline_flux():
    global pipe_txt2img_flux

    if pipe_txt2img_flux is not None:
        return pipe_txt2img_flux

    if FluxPipeline is None:
        raise RuntimeError(
            "FluxPipeline no está disponible con tu versión de diffusers. "
            "Sube diffusers a una versión que incluya FluxPipeline."
        )

    # FLUX recomienda bfloat16 en GPU modernas.
    # Si tu GPU/driver no soporta bien bf16, cambia a float16 en env: ISE_FLUX_DTYPE=float16
    flux_dtype_env = (os.getenv("ISE_FLUX_DTYPE", "bfloat16") or "bfloat16").strip().lower()
    if flux_dtype_env == "float16":
        flux_dtype = torch.float16
    else:
        flux_dtype = torch.bfloat16

    pipe_txt2img_flux = FluxPipeline.from_pretrained(
        HF_MODEL_ID_FLUX,
        torch_dtype=flux_dtype if DEVICE == "cuda" else torch.float32,
        cache_dir=str(MODELS_DIR),
    )

    if DEVICE == "cuda":
        pipe_txt2img_flux = pipe_txt2img_flux.to("cuda")

    # Opcional: reduce picos de VRAM
    try:
        pipe_txt2img_flux.enable_attention_slicing()
    except Exception:
        pass

    return pipe_txt2img_flux


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
# TXT2IMG (SD o FLUX) - MISMA FIRMA DE ENTRADA/SALIDA
# ---------------------------------------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = input_data.get("prompt", "") or ""
    negative_prompt = input_data.get("negative_prompt", "") or ""

    width, height = ensure_size_for_sd(
        int(input_data.get("width", 512)),
        int(input_data.get("height", 512)),
    )

    # Defaults: SD(22 pasos) / FLUX(4 pasos)
    steps = int(input_data.get("steps", 22))

    # SD guidance default 7.5
    # FLUX guidance suele ser 0.0 (schnell) según card
    guidance = float(input_data.get("guidance_scale", 7.5))

    provider = (input_data.get("model_provider") or MODEL_PROVIDER or "sd").strip().lower()

    if provider == "flux":
        pipe = get_txt2img_pipeline_flux()

        # Forzamos defaults sanos para FLUX schnell
        if steps <= 0:
            steps = 4
        if "guidance_scale" not in input_data:
            guidance = 0.0  # recomendado en schnell

        # max_sequence_length para prompts largos (default 256)
        max_seq = int(input_data.get("max_sequence_length", 256))

        # Seed opcional
        seed = input_data.get("seed", None)
        generator = None
        if seed is not None:
            try:
                seed_int = int(seed)
                generator = torch.Generator(device="cpu").manual_seed(seed_int)
            except Exception:
                generator = None

        with torch.autocast("cuda") if DEVICE == "cuda" else torch.no_grad():
            out = pipe(
                prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                max_sequence_length=max_seq,
                generator=generator,
            )

        img = out.images[0]
        file_path = _maybe_save_output(img, "isabelaos_flux")

        return {
            "image_b64": encode_image_to_b64(img),
            "file_path": file_path,   # "" por defecto (no guardamos)
            "mode": "txt2img",
            "provider": "flux",
            "steps": steps,
            "guidance_scale": guidance,
        }

    # default: SD (Realistic Vision)
    pipe = get_txt2img_pipeline_sd()

    if steps <= 0:
        steps = 22
    if guidance <= 0:
        guidance = 7.5

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
    file_path = _maybe_save_output(img, "isabelaos_sd")

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": file_path,  # "" por defecto (no guardamos)
        "mode": "txt2img",
        "provider": "sd",
        "steps": steps,
        "guidance_scale": guidance,
    }

# ---------------------------------------------------------
# NAVIDAD ESTUDIO (ORIGINAL – SIN TOCAR) ✅
# Nota: esto sigue usando SD-inpainting (no Flux) porque Flux no es inpaint aquí.
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
            num_inference_steps=28,
            guidance_scale=7.5,
        )

    final_image = result.images[0]
    file_path = _maybe_save_output(final_image, "isabelaos_navidad_estudio")

    return {
        "image_b64": encode_image_to_b64(final_image),
        "file_path": file_path,
        "mode": "navidad_estudio",
    }

# ---------------------------------------------------------
# HEADSHOT PRO (ORIGINAL) ✅
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
    file_path = _maybe_save_output(img, "isabelaos_headshot")

    return {
        "image_b64": encode_image_to_b64(img),
        "file_path": file_path,
        "mode": "headshot_pro",
        "style": style,
    }

# ---------------------------------------------------------
# HANDLER PRINCIPAL (MISMA LÓGICA)
# ---------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = input_data.get("action")

        if action == "health":
            return {
                "message": "isabelaOs worker online",
                "provider_default": MODEL_PROVIDER,
                "device": DEVICE,
                "volume_mount": str(BASE_DIR),
                "save_outputs": SAVE_OUTPUTS,
            }

        if action == "navidad_estudio":
            return handle_navidad_estudio(input_data)

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        # default: txt2img
        return handle_txt2img(input_data)

    except Exception as e:
        print("[rp_handler] ERROR:", repr(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
