# rp_handler.py – IsabelaOS Studio
# FLUX txt2img + SDXL img2img Product Studio + SDXL img2img Anime Identity
# + Prompt libre por usuario
# + Avatar support:
#   - effective_prompt
#   - avatar_id
#   - avatar_trigger
#   - avatar_lora_path
#   - intento de cargar LoRA desde Supabase Storage con cache local
#
# NOTA:
# - Si el LoRA falla, NO rompe el render.
# - Hace fallback a prompt + trigger solamente.

import os
import io
import json
import base64
import urllib.request
import urllib.parse
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

# cache local para LoRAs de avatar
LORA_CACHE_DIR = f"{BASE_VOLUME}/avatar_loras"
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

from diffusers import FluxPipeline, AutoPipelineForImage2Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE_FLUX = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_SDXL = (
    torch.bfloat16
    if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else (torch.float16 if DEVICE == "cuda" else torch.float32)
)

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_IMG2IMG_ID = os.environ.get("SDXL_IMG2IMG_ID", "stabilityai/stable-diffusion-xl-base-1.0")

flux_pipe: Optional[FluxPipeline] = None
img2img_pipe = None

# estado de LoRA actual cargado en FLUX
current_flux_lora_path: Optional[str] = None
current_flux_adapter_name: Optional[str] = None


def _set_torch_tweaks():
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_set_torch_tweaks()


# ----------------------------
# Env helpers
# ----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_AVATAR_BUCKET = os.environ.get("SUPABASE_AVATAR_BUCKET", os.environ.get("AVATAR_BUCKET", "avatars"))


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

    try:
        img2img_pipe.safety_checker = None
        img2img_pipe.requires_safety_checker = False
    except Exception as e:
        print("[IsabelaOS] Could not disable safety checker:", repr(e))

    if DEVICE == "cuda":
        img2img_pipe = img2img_pipe.to("cuda")

        try:
            if hasattr(img2img_pipe, "vae") and img2img_pipe.vae is not None:
                img2img_pipe.vae.to(dtype=torch.float32)
                print("[IsabelaOS] SDXL VAE forced to float32 ✅")
        except Exception as e:
            print("[IsabelaOS] Could not force VAE float32:", repr(e))

        try:
            img2img_pipe.enable_vae_slicing()
        except Exception:
            pass

    return img2img_pipe

# ----------------------------
# Helpers
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
    try:
        import numpy as np
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr.std() < 2.0
    except Exception:
        return False


def _safe_text(s: Any, max_len: int = 1200) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "").strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _normalize_storage_path(path: str) -> str:
    p = _safe_text(path, max_len=2000).lstrip("/")
    bucket_prefix = f"{SUPABASE_AVATAR_BUCKET}/"
    if p.startswith(bucket_prefix):
        p = p[len(bucket_prefix):]
    return p


def _make_local_lora_cache_path(storage_path: str) -> str:
    normalized = _normalize_storage_path(storage_path)
    safe_name = normalized.replace("/", "__")
    return os.path.join(LORA_CACHE_DIR, safe_name)


def _create_supabase_signed_download_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Crea signed URL usando la API REST de Supabase Storage.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in worker env")

    normalized_path = _normalize_storage_path(storage_path)
    encoded_path = urllib.parse.quote(normalized_path, safe="/")

    url = f"{SUPABASE_URL}/storage/v1/object/sign/{SUPABASE_AVATAR_BUCKET}/{encoded_path}"

    payload = json.dumps({"expiresIn": expires_in}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)

    signed_part = data.get("signedURL") or data.get("signedUrl")
    if not signed_part:
        raise RuntimeError(f"Could not create signed URL for {normalized_path}: {data}")

    if signed_part.startswith("http://") or signed_part.startswith("https://"):
        return signed_part

    return f"{SUPABASE_URL}/storage/v1{signed_part}"


def _download_avatar_lora_to_cache(storage_path: str) -> str:
    """
    Descarga el .safetensors desde Supabase a cache local del worker.
    """
    local_path = _make_local_lora_cache_path(storage_path)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"[IsabelaOS] Avatar LoRA already cached: {local_path}")
        return local_path

    signed_url = _create_supabase_signed_download_url(storage_path)
    print(f"[IsabelaOS] Downloading avatar LoRA from Supabase: {storage_path}")

    tmp_path = local_path + ".tmp"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with urllib.request.urlopen(signed_url, timeout=120) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp_path, local_path)
    print(f"[IsabelaOS] Avatar LoRA cached at: {local_path}")
    return local_path


def _unload_flux_lora_if_any(pipe) -> None:
    global current_flux_lora_path, current_flux_adapter_name

    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
            print("[IsabelaOS] Previous FLUX LoRA unloaded")
    except Exception as e:
        print("[IsabelaOS] Could not unload previous FLUX LoRA:", repr(e))

    current_flux_lora_path = None
    current_flux_adapter_name = None


def _ensure_flux_avatar_lora(pipe, avatar_lora_path: Optional[str], avatar_id: Optional[str]) -> Dict[str, Any]:
    """
    Intenta cargar el LoRA del avatar en FLUX.
    Si falla, devuelve info de warning pero no rompe el render.
    """
    global current_flux_lora_path, current_flux_adapter_name

    if not avatar_lora_path:
        if current_flux_lora_path:
            _unload_flux_lora_if_any(pipe)
        return {"used_lora": False, "warning": None}

    try:
        local_lora_file = _download_avatar_lora_to_cache(avatar_lora_path)

        # si ya está cargado el mismo, no recargar
        if current_flux_lora_path == local_lora_file:
            print("[IsabelaOS] Same avatar LoRA already loaded in FLUX")
            return {"used_lora": True, "warning": None}

        # descargar/cargar nuevo -> descargamos el anterior si había
        if current_flux_lora_path and current_flux_lora_path != local_lora_file:
            _unload_flux_lora_if_any(pipe)

        adapter_name = f"avatar_{avatar_id or 'default'}"

        print(f"[IsabelaOS] Loading avatar LoRA into FLUX: {local_lora_file}")

        # diffusers suele trabajar mejor con directorio + weight_name
        pipe.load_lora_weights(
            os.path.dirname(local_lora_file),
            weight_name=os.path.basename(local_lora_file),
            adapter_name=adapter_name,
        )

        # activar adapter si existe método
        try:
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        except Exception as e:
            print("[IsabelaOS] Could not set adapter weights:", repr(e))

        current_flux_lora_path = local_lora_file
        current_flux_adapter_name = adapter_name

        print("[IsabelaOS] Avatar LoRA loaded successfully")
        return {"used_lora": True, "warning": None}

    except Exception as e:
        warn = f"AVATAR_LORA_LOAD_FAILED: {e}"
        print("[IsabelaOS] WARNING:", warn)
        return {"used_lora": False, "warning": warn}


# ----------------------------
# Actions
# ----------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_flux()

    # prompt base y prompt efectivo
    prompt = _safe_text(input_data.get("prompt", ""))
    effective_prompt = _safe_text(input_data.get("effective_prompt", "")) or prompt
    negative_prompt = _safe_text(input_data.get("negative_prompt", ""))

    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    avatar_id = _safe_text(input_data.get("avatar_id", "")) or None
    avatar_name = _safe_text(input_data.get("avatar_name", "")) or None
    avatar_trigger = _safe_text(input_data.get("avatar_trigger", "")) or None
    avatar_lora_path = _safe_text(input_data.get("avatar_lora_path", "")) or None

    # intentar cargar LoRA del avatar si viene
    lora_info = _ensure_flux_avatar_lora(pipe, avatar_lora_path, avatar_id)

    print(
        "[txt2img_flux]",
        {
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "avatar_id": avatar_id,
            "avatar_name": avatar_name,
            "avatar_trigger": avatar_trigger,
            "avatar_lora_path": avatar_lora_path,
            "used_lora": lora_info.get("used_lora"),
        },
    )

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=DTYPE_FLUX):
                image = pipe(
                    prompt=effective_prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                ).images[0]
        else:
            image = pipe(
                prompt=effective_prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
            ).images[0]

    enc = encode_image_jpg(image)
    return {
        **enc,
        "mode": "txt2img_flux",
        "engine": "flux",
        "warning": lora_info.get("warning"),
        "avatar": {
            "id": avatar_id,
            "name": avatar_name,
            "trigger": avatar_trigger,
            "lora_path": avatar_lora_path,
            "used_lora": lora_info.get("used_lora", False),
        },
        "params": {
            "steps": steps,
            "size": [width, height],
            "used_effective_prompt": bool(effective_prompt),
        },
    }


def handle_product_studio_premium(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_img2img()

    if not input_data.get("image_b64"):
        return {"error": "MISSING_IMAGE_B64"}

    init_img = decode_image(input_data["image_b64"])
    init_img = clamp_size(init_img, max_side=int(input_data.get("max_side", 768)))
    w, h = init_img.size

    user_prompt = _safe_text(input_data.get("prompt"))
    user_negative = _safe_text(input_data.get("negative_prompt"))

    default_prompt = (
        "commercial product photography, professional studio lighting, softbox lighting, "
        "soft natural shadow under the product, clean seamless white background, "
        "high-end e-commerce photo, realistic texture detail, sharp focus, color accurate, "
        "premium advertising photo, minimal composition"
    )
    default_negative = (
        "text, watermark, logo, extra objects, clutter, messy background, "
        "low quality, blurry, distorted shape, oversharpen, cartoon, anime, unrealistic lighting"
    )

    prompt = user_prompt if user_prompt else default_prompt
    negative = default_negative + (", " + user_negative if user_negative else "")

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
        f"[product_studio_premium] size={w}x{h} steps={steps} guidance={guidance} strength={strength} dtype={DTYPE_SDXL} "
        f"prompt_user={'yes' if bool(user_prompt) else 'no'}"
    )

    with torch.inference_mode():
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
        "mode": "product_studio_premium",
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
            "used_user_prompt": bool(user_prompt),
        },
    }


def handle_transform_anime_identity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipe = get_img2img()

    if not input_data.get("image_b64"):
        return {"error": "MISSING_IMAGE_B64"}

    init_img = decode_image(input_data["image_b64"])
    init_img = clamp_size(init_img, max_side=int(input_data.get("max_side", 768)))
    w, h = init_img.size

    user_prompt = _safe_text(input_data.get("prompt"))
    user_negative = _safe_text(input_data.get("negative_prompt"))

    default_prompt = (
        "high detail anime portrait, cinematic lighting, dramatic rim light, "
        "sharp eyes, preserve facial identity, preserve facial proportions, "
        "same facial structure, same expression, same hairstyle, "
        "anime style but realistic proportions, clean high-quality render, "
        "soft glow, ultra detailed face, smooth skin shading, "
        "dynamic colorful background, studio quality, trending anime art style"
    )

    default_negative = (
        "different person, unrecognizable face, identity change, face swap, "
        "deformed face, distorted features, asymmetrical eyes, extra eyes, "
        "bad anatomy, low quality, blurry, jpeg artifacts, "
        "creepy, melted face, warped head, "
        "text, watermark, logo"
    )

    prompt = user_prompt if user_prompt else default_prompt
    prompt = prompt + ", same person, preserve identity, same face, same facial structure"
    negative = default_negative + (", " + user_negative if user_negative else "")

    steps = int(input_data.get("steps", 32))
    guidance = float(input_data.get("guidance", 7.5))
    strength = float(input_data.get("strength", 0.55))
    seed = input_data.get("seed", None)

    generator = None
    if seed is not None:
        try:
            seed = int(seed)
            generator = torch.Generator(device=("cuda" if DEVICE == "cuda" else "cpu")).manual_seed(seed)
        except Exception:
            generator = None

    print(
        f"[anime_identity] size={w}x{h} steps={steps} guidance={guidance} strength={strength} dtype={DTYPE_SDXL} "
        f"prompt_user={'yes' if bool(user_prompt) else 'no'}"
    )

    with torch.inference_mode():
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
        "mode": "transform_anime_identity",
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
            "used_user_prompt": bool(user_prompt),
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
            return {"message": "IsabelaOS worker online (FLUX txt2img + SDXL img2img Product + Anime Identity + Avatar support)"}

        if action == "headshot_pro":
            return handle_product_studio_premium(input_data)

        if action == "transform_anime_identity":
            return handle_transform_anime_identity(input_data)

        return handle_txt2img(input_data)

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
