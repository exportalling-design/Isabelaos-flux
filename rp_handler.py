# rp_handler.py – IsabelaOS Studio (FLUX txt2img + SDXL img2img Product Studio Premium)

import os
import io
import base64
from typing import Dict, Any, Optional

import torch
from PIL import Image
import runpod

# ----------------------------
# Cache paths (RunPod Volume)
# ----------------------------
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

# ----------------------------
# Diffusers
# ----------------------------
from diffusers import FluxPipeline, AutoPipelineForImage2Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FLUX suele ir bien en fp16
DTYPE_FLUX = torch.float16 if DEVICE == "cuda" else torch.float32

# SDXL más estable en bf16 (A100 soporta), si no, fp16.
DTYPE_SDXL = (
    torch.bfloat16
    if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else (torch.float16 if DEVICE == "cuda" else torch.float32)
)

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_IMG2IMG_ID = os.environ.get("SDXL_IMG2IMG_ID", "stabilityai/stable-diffusion-xl-base-1.0")

flux_pipe: Optional[FluxPipeline] = None
img2img_pipe = None


def _set_torch_tweaks():
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_set_torch_tweaks()


# ----------------------------
# Pipelines
# ----------------------------
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

    # Safety checker OFF (evita outputs negros por filtros)
    try:
        img2img_pipe.safety_checker = None
        img2img_pipe.requires_safety_checker = False
    except Exception as e:
        print("[IsabelaOS] Could not disable safety checker:", repr(e))

    if DEVICE == "cuda":
        img2img_pipe = img2img_pipe.to("cuda")

        # ✅ CLAVE para evitar outputs grises/NaN: VAE en float32
        try:
            if hasattr(img2img_pipe, "vae") and img2img_pipe.vae is not None:
                img2img_pipe.vae.to(dtype=torch.float32)
                print("[IsabelaOS] SDXL VAE forced to float32 ✅")
        except Exception as e:
            print("[IsabelaOS] Could not force VAE float32:", repr(e))

        # Opcional (ahorra VRAM y puede ayudar estabilidad)
        try:
            img2img_pipe.enable_vae_slicing()
        except Exception:
            pass

    return img2img_pipe


# ----------------------------
# Helpers (base64 <-> PIL)
# ----------------------------
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
    return Image.open(io.BytesIO(raw)).convert("RGB")


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
    # Detecta outputs planos (gris/negro)
    try:
        import numpy as np

        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr.std() < 2.0
    except Exception:
        return False


# ----------------------------
# Actions
# ----------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux()

    prompt = input_data.get("prompt", "")
    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=DTYPE_FLUX):
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

    enc = encode_image_jpg(image)
    return {
        **enc,
        "mode": "txt2img_flux",
        "engine": "flux",
        "params": {"steps": steps, "size": [width, height]},
    }


def handle_headshot_pro(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ Este action lo dejamos como "headshot_pro" para no cambiar tu backend ahora,
    pero en realidad es PRODUCT STUDIO PREMIUM.
    """
    pipe = get_img2img()

    if not input_data.get("image_b64"):
        return {"error": "MISSING_IMAGE_B64"}

    init_img = decode_image(input_data["image_b64"])
    init_img = clamp_size(init_img, max_side=int(input_data.get("max_side", 768)))
    w, h = init_img.size

    # ---------- Premium Product Studio Prompt ----------
    prompt = (
        "commercial product photography, professional studio lighting, softbox lighting, "
        "soft natural shadow under the product, clean seamless white background, "
        "high-end e-commerce photo, realistic texture detail, sharp focus, color accurate, "
        "premium advertising photo, minimal composition"
    )

    negative = (
        "text, watermark, logo, people, face, hands, extra objects, clutter, messy background, "
        "low quality, blurry, distorted shape, oversharpen, cartoon, anime, unrealistic lighting"
    )

    # Defaults premium (pero puedes sobre-escribir desde frontend si quieres)
    steps = int(input_data.get("steps", 30))
    guidance = float(input_data.get("guidance", 6.5))
    strength = float(input_data.get("strength", 0.38))
    seed = input_data.get("seed", None)

    generator = None
    if seed is not None:
        try:
            seed = int(seed)
            generator = torch.Generator(device=("cuda" if DEVICE == "cuda" else "cpu")).manual_seed(seed)
        except Exception:
            generator = None

    print(
        f"[product_studio_premium] size={w}x{h} steps={steps} guidance={guidance} strength={strength} dtype={DTYPE_SDXL}"
    )

    with torch.inference_mode():
        if DEVICE == "cuda":
            # ✅ Sin autocast aquí (reduce NaNs/gris)
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

    warning = None
    if is_flat_or_suspicious(out):
        warning = "SUSPICIOUS_FLAT_OUTPUT_FALLBACK_TO_INIT"
        print("[IsabelaOS] WARNING: flat output detected; returning init image fallback.")
        out = init_img

    enc = encode_image_jpg(out)
    return {
        **enc,
        "mode": "img2img_product_studio_premium",
        "engine": "sdxl_img2img",
        "warning": warning,
        "params": {
            "steps": steps,
            "guidance": guidance,
            "strength": strength,
            "seed": seed,
            "size": [w, h],
            "dtype_sdxl": str(DTYPE_SDXL),
            "vae_fp32": True,
        },
    }


# ----------------------------
# Main handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = (input_data.get("action") or "").strip()
        print("[IsabelaOS] action =", action or "(empty)")

        if action == "health":
            return {"message": "IsabelaOS worker online (FLUX txt2img + SDXL img2img Product Studio Premium)"}

        if action == "headshot_pro":
            return handle_headshot_pro(input_data)

        # default: FLUX txt2img
        return handle_txt2img(input_data)

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
