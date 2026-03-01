# rp_handler.py – IsabelaOS Studio (FLUX + SDXL IMG2IMG in ONE Serverless Worker)
import os
import io
import base64
from typing import Dict, Any, Optional

import torch
from PIL import Image
import runpod

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

from diffusers import FluxPipeline, AutoPipelineForImage2Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ DTYPE: FLUX ok en fp16; SDXL img2img mejor en bf16 en A100
DTYPE_FLUX = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_SDXL = (torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16) if DEVICE == "cuda" else torch.float32

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_IMG2IMG_ID = os.environ.get("SDXL_IMG2IMG_ID", "stabilityai/stable-diffusion-xl-base-1.0")

flux_pipe: Optional[FluxPipeline] = None
img2img_pipe = None


def _set_torch_tweaks():
    if DEVICE == "cuda":
        # ayudan a estabilidad/perf
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_set_torch_tweaks()


def get_flux() -> FluxPipeline:
    global flux_pipe
    if flux_pipe is not None:
        return flux_pipe

    print("[IsabelaOS] Loading FLUX pipeline...")
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE_FLUX,
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
        torch_dtype=DTYPE_SDXL,
        cache_dir=os.environ["HF_HUB_CACHE"],
        use_safetensors=True,
    )

    # ✅ Safety checker off (no negro por filtros)
    try:
        img2img_pipe.safety_checker = None
        img2img_pipe.requires_safety_checker = False
    except Exception as e:
        print("[IsabelaOS] Could not disable safety checker:", repr(e))

    if DEVICE == "cuda":
        img2img_pipe = img2img_pipe.to("cuda")

        # ✅ CLAVE: VAE en float32 para evitar NaNs/gray output
        try:
            if hasattr(img2img_pipe, "vae") and img2img_pipe.vae is not None:
                img2img_pipe.vae.to(dtype=torch.float32)
                print("[IsabelaOS] SDXL VAE forced to float32 ✅")
        except Exception as e:
            print("[IsabelaOS] Could not force VAE float32:", repr(e))

        # opcional, ayuda VRAM/estabilidad
        try:
            img2img_pipe.enable_vae_slicing()
        except Exception:
            pass

    return img2img_pipe


def encode_image_jpg(img: Image.Image, quality: int = 92) -> Dict[str, str]:
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "image_b64": b64,
        "image_data_url": "data:image/jpeg;base64," + b64,
        "mime": "image/jpeg",
    }


def decode_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def clamp_size(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw = int((w * scale) // 8 * 8)
    nh = int((h * scale) // 8 * 8)
    nw = max(nw, 256)
    nh = max(nh, 256)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def is_flat_or_suspicious(img: Image.Image) -> bool:
    # detecta imágenes “planas” (gris/negro) típicas de NaNs
    try:
        import numpy as np
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        # si casi no hay variación -> sospechoso
        return (arr.std() < 2.0)
    except Exception:
        return False


def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux()
    prompt = input_data.get("prompt", "")
    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    with torch.inference_mode():
        if DEVICE == "cuda":
            # FLUX ok con autocast fp16
            with torch.autocast("cuda", dtype=DTYPE_FLUX):
                image = pipe(prompt=prompt, num_inference_steps=steps, width=width, height=height).images[0]
        else:
            image = pipe(prompt=prompt, num_inference_steps=steps, width=width, height=height).images[0]

    enc = encode_image_jpg(image)
    return {**enc, "mode": "txt2img_flux", "engine": "flux"}


def handle_headshot_pro(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_img2img()

    if not input_data.get("image_b64"):
        return {"error": "MISSING_IMAGE_B64"}

    init_img = decode_image(input_data["image_b64"])
    init_img = clamp_size(init_img, max_side=int(input_data.get("max_side", 768)))
    w, h = init_img.size

    user_style = (input_data.get("style") or "corporate").strip().lower()

    if user_style == "creative":
        prompt = (
            "professional portrait photo, studio lighting, softbox, clean background, "
            "high detail, realistic skin texture, sharp focus, premium look"
        )
    elif user_style == "influencer":
        prompt = (
            "portrait photo, natural soft lighting, clean background, "
            "sharp focus, realistic, premium social media headshot"
        )
    else:
        prompt = (
            "professional corporate headshot portrait, studio lighting, soft shadows, "
            "clean seamless background, sharp focus, realistic, high detail"
        )

    negative = (
        "text, logo, watermark, deformed face, extra limbs, blurry, low quality, "
        "cartoon, anime, oversaturated, artifacts"
    )

    steps = int(input_data.get("steps", 20))
    guidance = float(input_data.get("guidance", 5.0))
    strength = float(input_data.get("strength", 0.20))
    seed = input_data.get("seed", None)

    generator = None
    if seed is not None:
        try:
            seed = int(seed)
            generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)
        except Exception:
            generator = None

    print(f"[headshot_pro] size={w}x{h} steps={steps} guidance={guidance} strength={strength} style={user_style}")

    with torch.inference_mode():
        if DEVICE == "cuda":
            # ✅ IMPORTANTE: NO autocast aquí (reduce NaNs con SDXL)
            out = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                width=w,
                height=h,
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
                width=w,
                height=h,
                generator=generator,
            ).images[0]

    # ✅ si salió “plano gris/negro”, devolver init_img y marcar warning
    warning = None
    if is_flat_or_suspicious(out):
        warning = "SUSPICIOUS_FLAT_OUTPUT"
        print("[IsabelaOS] WARNING: output looks flat (gray/black). Returning init image as fallback.")
        out = init_img

    enc = encode_image_jpg(out)
    return {
        **enc,
        "mode": "img2img_sdxl_headshot_pro",
        "engine": "sdxl_img2img",
        "warning": warning,
        "params": {
            "steps": steps,
            "guidance": guidance,
            "strength": strength,
            "seed": seed,
            "style": user_style,
            "size": [w, h],
            "dtype_sdxl": str(DTYPE_SDXL),
            "vae_fp32": True,
        },
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = (input_data.get("action") or "").strip()
        print("[IsabelaOS] action =", action or "(empty)")

        if action == "health":
            return {"message": "IsabelaOS worker online (FLUX txt2img + SDXL img2img headshot_pro)"}

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        return handle_txt2img(input_data)

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
