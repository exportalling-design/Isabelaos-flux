# rp_handler.py – IsabelaOS Studio
# FLUX txt2img + SDXL img2img Product Studio + SDXL img2img Anime Identity
# + Prompt libre por usuario
# + Avatar support
# + NUEVO: compose_scene (Montaje IA local con fondo exacto pixel por pixel)
#
# NOTAS:
# - NO se toca la lógica principal de FLUX que ya te funciona.
# - Se agrega un módulo nuevo local para montaje realista.
# - Si el usuario sube fondo, este se usa EXACTO, no se regenera.
# - El montaje hace:
#   1) recorte automático del sujeto
#   2) escalado / posición
#   3) feather de bordes
#   4) color match básico
#   5) blend seamless o alpha
#
# Requiere además:
#   pip install numpy opencv-python-headless rembg onnxruntime

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

# NUEVO para montaje IA
import numpy as np
import cv2
from rembg import remove as rembg_remove

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
# Helpers generales
# ----------------------------
def encode_image_jpg(img: Image.Image, quality: int = 92) -> Dict[str, str]:
    """
    Devuelve formato principal + aliases legacy para no romper
    endpoints viejos que busquen otras llaves.
    """
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64

    return {
        # llaves principales
        "image_b64": b64,
        "image_data_url": data_url,
        "mime": "image/jpeg",

        # aliases legacy
        "result_b64": b64,
        "resultBase64": b64,
        "image": b64,
        "image_base64": b64,
        "data_url": data_url,
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


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _safe_int(v, d=0):
    try:
        return int(v)
    except Exception:
        return d


def _clamp(x, a, b):
    return max(a, min(b, x))

# ----------------------------
# Helpers avatar LoRA
# ----------------------------
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
    global current_flux_lora_path, current_flux_adapter_name

    if not avatar_lora_path:
        if current_flux_lora_path:
            _unload_flux_lora_if_any(pipe)
        return {"used_lora": False, "warning": None}

    try:
        local_lora_file = _download_avatar_lora_to_cache(avatar_lora_path)

        if current_flux_lora_path == local_lora_file:
            print("[IsabelaOS] Same avatar LoRA already loaded in FLUX")
            return {"used_lora": True, "warning": None}

        if current_flux_lora_path and current_flux_lora_path != local_lora_file:
            _unload_flux_lora_if_any(pipe)

        adapter_name = f"avatar_{avatar_id or 'default'}"

        print(f"[IsabelaOS] Loading avatar LoRA into FLUX: {local_lora_file}")

        pipe.load_lora_weights(
            os.path.dirname(local_lora_file),
            weight_name=os.path.basename(local_lora_file),
            adapter_name=adapter_name,
        )

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
# Helpers montaje IA
# ----------------------------
def _feather_alpha(alpha: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return alpha
    k = feather_px * 2 + 1
    k = max(3, k)
    return cv2.GaussianBlur(alpha, (k, k), 0)


def _match_color_simple(fg_bgr: np.ndarray, bg_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ajuste simple de media/std por canal usando la zona de fondo cercana.
    No cambia identidad, solo ayuda a integrar color/luz.
    """
    m = (mask > 0)
    if m.sum() < 50:
        return fg_bgr

    fg = fg_bgr.astype(np.float32)
    bg = bg_bgr.astype(np.float32)

    out = fg.copy()
    for c in range(3):
        fg_vals = fg[..., c][m]

        kernel = np.ones((31, 31), np.uint8)
        ring = cv2.dilate(mask, kernel, iterations=1) > 0
        bg_vals = bg[..., c][ring]

        if bg_vals.size < 50:
            continue

        fg_mean, fg_std = fg_vals.mean(), fg_vals.std() + 1e-6
        bg_mean, bg_std = bg_vals.mean(), bg_vals.std() + 1e-6

        out[..., c] = (out[..., c] - fg_mean) * (bg_std / fg_std) + bg_mean

    return np.clip(out, 0, 255).astype(np.uint8)


def _add_contact_shadow(bg_bgr: np.ndarray, mask_roi: np.ndarray, roi_box, opacity: float = 0.18) -> np.ndarray:
    """
    Sombra de contacto simple bajo el sujeto.
    Muy útil para quitar el look de "pegado".
    """
    x1, y1, x2, y2 = roi_box
    out = bg_bgr.copy()

    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return out

    # Base shadow desde mask
    shadow = (mask_roi > 10).astype(np.uint8) * 255
    shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_LINEAR)

    # Comprimir verticalmente la sombra para que quede debajo de pies/base
    compressed_h = max(8, int(h * 0.18))
    shadow_small = cv2.resize(shadow, (w, compressed_h), interpolation=cv2.INTER_AREA)

    # suavizar
    shadow_small = cv2.GaussianBlur(shadow_small, (0, 0), sigmaX=9, sigmaY=5)

    # crear canvas de sombra
    shadow_canvas = np.zeros((bg_bgr.shape[0], bg_bgr.shape[1]), dtype=np.float32)

    sy1 = min(bg_bgr.shape[0] - 1, max(0, y2 - compressed_h // 2))
    sy2 = min(bg_bgr.shape[0], sy1 + compressed_h)
    sx1 = max(0, x1)
    sx2 = min(bg_bgr.shape[1], x2)

    if sy2 > sy1 and sx2 > sx1:
        crop = shadow_small[: sy2 - sy1, : sx2 - sx1].astype(np.float32) / 255.0
        shadow_canvas[sy1:sy2, sx1:sx2] = crop

    shadow_canvas = cv2.GaussianBlur(shadow_canvas, (0, 0), sigmaX=12, sigmaY=8)
    shadow_canvas = np.clip(shadow_canvas * opacity, 0.0, 1.0)

    # oscurecer fondo
    for c in range(3):
        out[..., c] = out[..., c].astype(np.float32) * (1.0 - shadow_canvas)

    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------
# Actions
# ----------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    FLUX txt2img.
    ESTA PARTE SE DEJA INTACTA, solo con tus mejoras actuales.
    """
    pipe = get_flux()

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


def handle_compose_scene(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    NUEVO MÓDULO: MONTAJE IA LOCAL
    Usa el fondo EXACTO subido por el usuario.
    No regenera el escenario.
    No usa Flux.
    No cambia la identidad del sujeto.
    """
    fg_b64 = input_data.get("fg_image_b64") or input_data.get("person_image")
    bg_b64 = input_data.get("bg_image_b64") or input_data.get("background_image")

    if not fg_b64 or not bg_b64:
        return {"error": "MISSING_FG_OR_BG"}

    # parámetros interpretados por Isabela o enviados por backend
    x = _clamp(_safe_float(input_data.get("x", 0.5), 0.5), 0.0, 1.0)
    y = _clamp(_safe_float(input_data.get("y", 0.72), 0.72), 0.0, 1.0)
    scale = _clamp(_safe_float(input_data.get("scale", 0.55), 0.55), 0.1, 2.0)
    feather = _clamp(_safe_int(input_data.get("feather", 12), 12), 0, 40)
    blend_mode = _safe_text(input_data.get("mode", "seamless")).lower() or "seamless"
    color_match = bool(input_data.get("color_match", True))
    add_shadow = bool(input_data.get("shadow", True))

    print(
        "[compose_scene]",
        {
            "x": x,
            "y": y,
            "scale": scale,
            "feather": feather,
            "blend_mode": blend_mode,
            "color_match": color_match,
            "shadow": add_shadow,
        },
    )

    # --- cargar imágenes ---
    fg_pil = decode_image(fg_b64).convert("RGBA")
    bg_pil = decode_image(bg_b64).convert("RGB")

    bg_bgr = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
    bg_h, bg_w = bg_bgr.shape[:2]

    # --- recorte del sujeto con rembg ---
    fg_rgba = rembg_remove(np.array(fg_pil)).astype(np.uint8)

    if fg_rgba.ndim != 3 or fg_rgba.shape[2] == 3:
        alpha = np.ones((fg_rgba.shape[0], fg_rgba.shape[1]), dtype=np.uint8) * 255
        fg_rgba = np.dstack([fg_rgba, alpha])

    # --- escalar sujeto respecto al fondo ---
    fg_h, fg_w = fg_rgba.shape[:2]
    new_w = max(8, int(bg_w * scale))
    ratio = new_w / max(1, fg_w)
    new_h = max(8, int(fg_h * ratio))

    fg_rgba = cv2.resize(fg_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # --- calcular posición centrada ---
    cx = int(bg_w * x)
    cy = int(bg_h * y)

    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # --- recorte seguro si se sale del canvas ---
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(bg_w, x2), min(bg_h, y2)

    fg_x1 = bx1 - x1
    fg_y1 = by1 - y1
    fg_x2 = fg_x1 + (bx2 - bx1)
    fg_y2 = fg_y1 + (by2 - by1)

    if bx2 <= bx1 or by2 <= by1:
        return {"error": "PLACEMENT_OUT_OF_BOUNDS"}

    fg_crop = fg_rgba[fg_y1:fg_y2, fg_x1:fg_x2]
    fg_rgb = fg_crop[..., :3]
    fg_a = fg_crop[..., 3]

    # --- feather en bordes ---
    fg_a = _feather_alpha(fg_a, feather)
    mask255 = fg_a.copy()

    bg_roi = bg_bgr[by1:by2, bx1:bx2].copy()

    # --- ajuste de color ---
    if color_match:
        fg_rgb = _match_color_simple(fg_rgb, bg_roi, mask255)

    # --- composición ---
    if blend_mode == "alpha":
        alpha_f = (fg_a.astype(np.float32) / 255.0)[..., None]
        comp = (fg_rgb.astype(np.float32) * alpha_f) + (bg_roi.astype(np.float32) * (1.0 - alpha_f))
        bg_bgr[by1:by2, bx1:bx2] = np.clip(comp, 0, 255).astype(np.uint8)
    else:
        # seamless = más realista si funciona
        center = (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2)

        full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        full_mask[by1:by2, bx1:bx2] = (fg_a > 10).astype(np.uint8) * 255

        src = np.zeros_like(bg_bgr)
        src[by1:by2, bx1:bx2] = fg_rgb

        try:
            bg_bgr = cv2.seamlessClone(src, bg_bgr, full_mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print("[compose_scene] seamlessClone failed, fallback alpha:", repr(e))
            alpha_f = (fg_a.astype(np.float32) / 255.0)[..., None]
            comp = (fg_rgb.astype(np.float32) * alpha_f) + (bg_roi.astype(np.float32) * (1.0 - alpha_f))
            bg_bgr[by1:by2, bx1:bx2] = np.clip(comp, 0, 255).astype(np.uint8)

    # --- sombra de contacto ---
    if add_shadow:
        bg_bgr = _add_contact_shadow(bg_bgr, mask255, (bx1, by1, bx2, by2), opacity=0.18)

    out_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out_rgb)

    enc = encode_image_jpg(out_pil)
    return {
        **enc,
        "mode": "compose_scene_v15",
        "engine": "local_compositor",
        "params": {
            "x": x,
            "y": y,
            "scale": scale,
            "feather": feather,
            "blend_mode": blend_mode,
            "color_match": color_match,
            "shadow": add_shadow,
        },
    }

    # ----------------------------
# Main handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or {}
        action = _safe_text(input_data.get("action", ""))
        print("[IsabelaOS] action =", action or "(empty)")

        if action == "health":
            return {
                "message": "IsabelaOS worker online (FLUX txt2img + SDXL img2img Product + Anime Identity + Avatar support + Compose Scene)"
            }

        # ✅ FLUX intacto
        if action == "txt2img_flux" or action == "generate_image" or action == "txt2img":
            return handle_txt2img(input_data)

        # ✅ módulos previos intactos
        if action == "headshot_pro":
            return handle_product_studio_premium(input_data)

        if action == "transform_anime_identity":
            return handle_transform_anime_identity(input_data)

        # ✅ NUEVO módulo
        if action == "compose_scene":
            return handle_compose_scene(input_data)

        return {"error": "UNKNOWN_ACTION", "action": action}

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

                                       
