# rp_handler.py – IsabelaOS Studio
# FLUX txt2img + anchors + identity lock (InsightFace / FaceSwap)
# + compose_scene (Montaje IA local)
# + skin_mode natural con Realistic Vision cuando hay avatar/anchor

import os
import io
import json
import time
import base64
import urllib.request
import traceback
import hashlib
from typing import Dict, Any, Optional, List

import cv2
import torch
import numpy as np
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

# cache local para anchors de avatar
ANCHOR_CACHE_DIR = f"{BASE_VOLUME}/avatar_anchors"
os.makedirs(ANCHOR_CACHE_DIR, exist_ok=True)

# cache local para modelos de face swap
FACE_MODELS_DIR = f"{BASE_VOLUME}/face_models"
os.makedirs(FACE_MODELS_DIR, exist_ok=True)

from diffusers import (
    FluxPipeline,
    AutoPipelineForImage2Image,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
    AutoencoderKL,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE_FLUX = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_SDXL = (
    torch.bfloat16
    if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else (torch.float16 if DEVICE == "cuda" else torch.float32)
)
DTYPE_SD15 = torch.float16 if DEVICE == "cuda" else torch.float32

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_IMG2IMG_ID = os.environ.get("SDXL_IMG2IMG_ID", "stabilityai/stable-diffusion-xl-base-1.0")

# Realistic Vision para skin_mode natural + avatar
REALISTIC_MODEL_ID = os.environ.get("REALISTIC_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE")
REALISTIC_VAE_ID = os.environ.get("REALISTIC_VAE_ID", "stabilityai/sd-vae-ft-mse-original")

# Modelo de face swap
INSWAPPER_MODEL_PATH = f"{FACE_MODELS_DIR}/inswapper_128.onnx"
INSWAPPER_MODEL_URL = os.environ.get(
    "INSWAPPER_MODEL_URL",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
)

flux_pipe: Optional[FluxPipeline] = None
img2img_pipe = None
realistic_pipe = None

# insightface (lazy load)
face_analyser = None
face_swapper = None


def _set_torch_tweaks():
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_set_torch_tweaks()

print("[IsabelaOS] Worker booting...")
print("[IsabelaOS] DEVICE =", DEVICE)
print("[IsabelaOS] DTYPE_FLUX =", DTYPE_FLUX)
print("[IsabelaOS] DTYPE_SDXL =", DTYPE_SDXL)
print("[IsabelaOS] DTYPE_SD15 =", DTYPE_SD15)
print("[IsabelaOS] BASE_VOLUME =", BASE_VOLUME)

# ----------------------------
# Pipelines
# ----------------------------
def get_flux() -> FluxPipeline:
    global flux_pipe

    if flux_pipe is not None:
        print("[IsabelaOS] FLUX already loaded ✅")
        return flux_pipe

    print("[IsabelaOS] Loading FLUX pipeline...")
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE_FLUX,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    print("[IsabelaOS] FLUX pipeline loaded from pretrained ✅")

    if DEVICE == "cuda":
        print("[IsabelaOS] Moving FLUX pipeline to CUDA...")
        flux_pipe = flux_pipe.to("cuda")
        print("[IsabelaOS] FLUX pipeline moved to CUDA ✅")

    return flux_pipe


def get_img2img():
    global img2img_pipe

    if img2img_pipe is not None:
        print("[IsabelaOS] SDXL IMG2IMG already loaded ✅")
        return img2img_pipe

    print("[IsabelaOS] Loading SDXL IMG2IMG pipeline...")
    img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
        SDXL_IMG2IMG_ID,
        torch_dtype=DTYPE_SDXL,
        cache_dir=os.environ["HF_HUB_CACHE"],
        use_safetensors=True,
    )
    print("[IsabelaOS] SDXL IMG2IMG loaded from pretrained ✅")

    try:
        img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(img2img_pipe.scheduler.config)
        print("[IsabelaOS] SDXL scheduler switched to UniPC ✅")
    except Exception as e:
        print("[IsabelaOS] Could not switch SDXL scheduler to UniPC:", repr(e))

    try:
        img2img_pipe.safety_checker = None
        img2img_pipe.requires_safety_checker = False
    except Exception as e:
        print("[IsabelaOS] Could not disable safety checker:", repr(e))

    if DEVICE == "cuda":
        print("[IsabelaOS] Moving SDXL IMG2IMG to CUDA...")
        img2img_pipe = img2img_pipe.to("cuda")
        print("[IsabelaOS] SDXL IMG2IMG moved to CUDA ✅")

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


def get_realistic_vision():
    global realistic_pipe

    if realistic_pipe is not None:
        print("[IsabelaOS] Realistic Vision already loaded ✅")
        return realistic_pipe

    print("[IsabelaOS] Loading Realistic Vision pipeline...")
    realistic_pipe = StableDiffusionPipeline.from_pretrained(
        REALISTIC_MODEL_ID,
        torch_dtype=DTYPE_SD15,
        cache_dir=os.environ["HF_HUB_CACHE"],
        safety_checker=None,
        requires_safety_checker=False,
    )
    print("[IsabelaOS] Realistic Vision loaded from pretrained ✅")

    try:
        print("[IsabelaOS] Loading Realistic Vision VAE...")
        vae = AutoencoderKL.from_pretrained(
            REALISTIC_VAE_ID,
            torch_dtype=DTYPE_SD15,
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        realistic_pipe.vae = vae
        print("[IsabelaOS] Realistic Vision VAE loaded ✅")
    except Exception as e:
        print("[IsabelaOS] Could not load Realistic Vision VAE:", repr(e))

    try:
        realistic_pipe.scheduler = UniPCMultistepScheduler.from_config(realistic_pipe.scheduler.config)
        print("[IsabelaOS] Realistic Vision scheduler switched to UniPC ✅")
    except Exception as e:
        print("[IsabelaOS] Could not switch Realistic Vision scheduler to UniPC:", repr(e))

    try:
        realistic_pipe.safety_checker = None
        realistic_pipe.requires_safety_checker = False
    except Exception as e:
        print("[IsabelaOS] Could not disable Realistic Vision safety checker:", repr(e))

    if DEVICE == "cuda":
        print("[IsabelaOS] Moving Realistic Vision to CUDA...")
        realistic_pipe = realistic_pipe.to("cuda")
        print("[IsabelaOS] Realistic Vision moved to CUDA ✅")

        try:
            realistic_pipe.enable_attention_slicing()
        except Exception:
            pass

    return realistic_pipe


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
        "image_b64": b64,
        "image_data_url": data_url,
        "mime": "image/jpeg",
        "result_b64": b64,
        "resultBase64": b64,
        "image": b64,
        "image_base64": b64,
        "data_url": data_url,
    }


def decode_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def clamp_size(img: Image.Image, max_side: int = 1024) -> Image.Image:
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


def _safe_list(v) -> List[str]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        s = _safe_text(item, max_len=4000)
        if s:
            out.append(s)
    return out


def pil_to_bgr(img: Image.Image):
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr):
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _add_subtle_sensor_noise(img: Image.Image, amount: float = 2.0) -> Image.Image:
    """
    Pequeño ruido para romper el look demasiado limpio.
    Muy sutil para no destruir identidad.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, amount, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Helpers avatar anchors
# ----------------------------
def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _guess_ext_from_url(url: str) -> str:
    lower = url.lower()
    if ".png" in lower:
        return "png"
    if ".webp" in lower:
        return "webp"
    if ".jpeg" in lower:
        return "jpeg"
    if ".jpg" in lower:
        return "jpg"
    return "jpg"


def _download_url_to_file(url: str, local_path: str) -> str:
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"[IsabelaOS] Avatar anchor already cached: {local_path}")
        return local_path

    tmp_path = local_path + ".tmp"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print("[IsabelaOS] Downloading avatar anchor from signed URL")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp_path, local_path)
    print(f"[IsabelaOS] Avatar anchor cached at: {local_path}")
    return local_path


def _cache_avatar_anchors(anchor_urls: List[str], avatar_id: Optional[str]) -> List[str]:
    cached = []

    for i, url in enumerate(anchor_urls[:3]):
        try:
            ext = _guess_ext_from_url(url)
            filename = f"{avatar_id or 'default'}_{i+1}_{_hash_url(url)}.{ext}"
            local_path = os.path.join(ANCHOR_CACHE_DIR, filename)
            cached_path = _download_url_to_file(url, local_path)
            cached.append(cached_path)
        except Exception as e:
            print("[IsabelaOS] WARNING: anchor download failed:", repr(e))

    return cached


def _load_anchor_images(anchor_urls: List[str], avatar_id: Optional[str]) -> List[Image.Image]:
    local_files = _cache_avatar_anchors(anchor_urls, avatar_id)
    images = []

    for path in local_files:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print("[IsabelaOS] WARNING: anchor image open failed:", repr(e))

    return images


# ----------------------------
# Helpers face lock / identity
# ----------------------------
def _ensure_file_from_url(url: str, local_path: str) -> str:
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    parent_dir = os.path.dirname(local_path) if local_path else ""
    if not parent_dir:
        parent_dir = FACE_MODELS_DIR

    os.makedirs(parent_dir, exist_ok=True)

    tmp_path = local_path + ".tmp"

    print(f"[IsabelaOS] Downloading model from: {url}")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp_path, local_path)
    print(f"[IsabelaOS] Model cached at: {local_path}")
    return local_path


def _get_ort_providers():
    if DEVICE == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_face_analyser():
    global face_analyser

    if face_analyser is not None:
        return face_analyser

    print("[IsabelaOS] Loading FaceAnalysis...")
    from insightface.app import FaceAnalysis

    face_analyser = FaceAnalysis(
        name="buffalo_l",
        root=FACE_MODELS_DIR,
        providers=_get_ort_providers(),
    )
    face_analyser.prepare(
        ctx_id=0 if DEVICE == "cuda" else -1,
        det_size=(640, 640),
    )
    print("[IsabelaOS] FaceAnalysis ready ✅")
    return face_analyser


def get_face_swapper():
    global face_swapper

    if face_swapper is not None:
        return face_swapper

    print("[IsabelaOS] Loading face swapper...")
    from insightface.model_zoo import get_model as insight_get_model

    os.makedirs(FACE_MODELS_DIR, exist_ok=True)

    model_path = INSWAPPER_MODEL_PATH or f"{FACE_MODELS_DIR}/inswapper_128.onnx"

    if INSWAPPER_MODEL_URL and not os.path.exists(model_path):
        _ensure_file_from_url(INSWAPPER_MODEL_URL, model_path)

    face_swapper = insight_get_model(
        model_path,
        providers=_get_ort_providers(),
    )
    print("[IsabelaOS] Face swapper ready ✅")
    return face_swapper


def _pick_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0, (x2 - x1)) * max(0, (y2 - y1))

    faces = sorted(faces, key=area, reverse=True)
    return faces[0]


def _apply_identity_lock(base_image: Image.Image, anchor_images: List[Image.Image]) -> (Image.Image, Optional[str]):
    """
    Hace face swap sobre la imagen generada usando la primera anchor válida.
    Se aplica SOLO si hay cara en anchor y en resultado.
    """
    if not anchor_images:
        return base_image, "IDENTITY_LOCK_SKIPPED_NO_ANCHORS"

    try:
        analyser = get_face_analyser()
        swapper = get_face_swapper()

        source_face = None
        for i, anchor in enumerate(anchor_images):
            anchor_bgr = pil_to_bgr(anchor)
            source_faces = analyser.get(anchor_bgr)
            source_face = _pick_largest_face(source_faces)
            if source_face is not None:
                print(f"[IsabelaOS] Source face found in anchor {i+1}")
                break

        if source_face is None:
            return base_image, "IDENTITY_LOCK_NO_FACE_IN_ANCHORS"

        target_bgr = pil_to_bgr(base_image)
        target_faces = analyser.get(target_bgr)
        target_face = _pick_largest_face(target_faces)

        if target_face is None:
            return base_image, "IDENTITY_LOCK_NO_FACE_IN_GENERATION"

        swapped_bgr = swapper.get(
            target_bgr,
            target_face,
            source_face,
            paste_back=True,
        )

        swapped_pil = bgr_to_pil(swapped_bgr)
        return swapped_pil, None

    except Exception as e:
        print("[IsabelaOS] WARNING: identity lock failed:", repr(e))
        print(traceback.format_exc())
        return base_image, f"IDENTITY_LOCK_FAILED: {e}"


# ----------------------------
# Helpers natural skin refine
# ----------------------------
def _refine_natural_skin(base_image: Image.Image, input_data: Dict[str, Any]) -> (Image.Image, Optional[str]):
    """
    Refine suave con SDXL para bajar el look embellecido / plástico
    sin reinterpretar demasiado el rostro.
    """
    try:
        pipe = get_img2img()

        orig_w, orig_h = base_image.size
        work_img = clamp_size(
            base_image,
            max_side=int(input_data.get("natural_skin_max_side", 1024))
        )

        # 1) Ruido MUY leve antes del refine para romper la piel "muñeca"
        work_img = _add_subtle_sensor_noise(
            work_img,
            amount=float(input_data.get("natural_skin_pre_noise", 1.6))
        )

        w, h = work_img.size

        refine_prompt = (
            "natural skin texture, visible pores, subtle skin imperfections, "
            "uneven skin tone, unretouched photo, preserve face identity, same face"
        )

        refine_negative = (
            "smooth skin, plastic skin, airbrushed skin, flawless skin, "
            "beauty filter, glossy skin, CGI, change face, different face"
        )

        steps = int(input_data.get("natural_skin_steps", 10))
        guidance = float(input_data.get("natural_skin_guidance", 2.2))
        strength = float(input_data.get("natural_skin_strength", 0.08))

        print(
            "[natural_skin_refine]",
            {
                "steps": steps,
                "guidance": guidance,
                "strength": strength,
                "size": [w, h],
            },
        )

        def _add_pre_refine_noise(img, amount=0.015):
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, amount * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        work_img = _add_pre_refine_noise(work_img, 0.015)

        with torch.inference_mode():
            out = pipe(
                prompt=refine_prompt,
                negative_prompt=refine_negative,
                image=work_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                width=w,
                height=h,
            ).images[0]

        if is_flat_or_suspicious(out):
            return base_image, "NATURAL_SKIN_FLAT_OUTPUT"

        if out.size != (orig_w, orig_h):
            out = out.resize((orig_w, orig_h), Image.LANCZOS)

        # 2) Ruido MUY leve final para romper el acabado demasiado limpio
        out = _add_subtle_sensor_noise(
            out,
            amount=float(input_data.get("natural_skin_post_noise", 1.2))
        )

        return out, None

    except Exception as e:
        print("[IsabelaOS] WARNING: natural skin refine failed:", repr(e))
        print(traceback.format_exc())
        return base_image, f"NATURAL_SKIN_FAILED: {e}"


# ----------------------------
# Helpers montaje IA
# ----------------------------
def _feather_alpha(alpha, feather_px: int):
    if feather_px <= 0:
        return alpha

    k = feather_px * 2 + 1
    k = max(3, k)
    return cv2.GaussianBlur(alpha, (k, k), 0)


def _match_color_simple(fg_bgr, bg_bgr, mask):
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


def _add_contact_shadow(bg_bgr, mask_roi, roi_box, opacity: float = 0.18):
    x1, y1, x2, y2 = roi_box
    out = bg_bgr.copy()

    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return out

    shadow = (mask_roi > 10).astype(np.uint8) * 255
    shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_LINEAR)

    compressed_h = max(8, int(h * 0.18))
    shadow_small = cv2.resize(shadow, (w, compressed_h), interpolation=cv2.INTER_AREA)
    shadow_small = cv2.GaussianBlur(shadow_small, (0, 0), sigmaX=9, sigmaY=5)

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

    for c in range(3):
        out[..., c] = out[..., c].astype(np.float32) * (1.0 - shadow_canvas)

    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------
# Actions
# ----------------------------
def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    print("[IsabelaOS] handle_txt2img() entered")

    prompt = _safe_text(input_data.get("prompt", ""))
    effective_prompt = _safe_text(input_data.get("effective_prompt", "")) or prompt
    negative_prompt = _safe_text(input_data.get("negative_prompt", ""))
    
    if "across frames" in effective_prompt.lower():
        effective_prompt = effective_prompt.replace("across frames", "")
    if "frame skipping" in effective_prompt.lower():
        effective_prompt = effective_prompt.replace("frame skipping", "")
    if "motion artifacts" in effective_prompt.lower():
        effective_prompt = effective_prompt.replace("motion artifacts", "")
    if "gentle blinking" in effective_prompt.lower():
        effective_prompt = effective_prompt.replace("gentle blinking", "")
    if "temporal wobble" in effective_prompt.lower():
        effective_prompt = effective_prompt.replace("temporal wobble", "")
        
    skin_mode = _safe_text(input_data.get("skin_mode", "standard")).lower() or "standard"

    steps = int(input_data.get("steps", 4))
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))

    avatar_id = _safe_text(input_data.get("avatar_id", "")) or None
    avatar_name = _safe_text(input_data.get("avatar_name", "")) or None

    avatar_anchor_urls = _safe_list(input_data.get("avatar_anchor_urls"))
    avatar_anchor_paths = _safe_list(input_data.get("avatar_anchor_paths"))

    # 1) Cargar anchors si existen
    anchor_images = []
    if avatar_anchor_urls:
        try:
            anchor_images = _load_anchor_images(avatar_anchor_urls, avatar_id)
        except Exception as e:
            print("[IsabelaOS] WARNING: anchor loading failed:", repr(e))

    has_avatar_anchor = bool(avatar_id and anchor_images)
    use_realistic_natural = bool(has_avatar_anchor and skin_mode == "natural")

    print(
        "[txt2img_pipeline]",
        {
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "negative_prompt": negative_prompt,
            "skin_mode": skin_mode,
            "steps": steps,
            "width": width,
            "height": height,
            "avatar_id": avatar_id,
            "avatar_name": avatar_name,
            "avatar_anchor_urls_count": len(avatar_anchor_urls),
            "avatar_anchor_paths_count": len(avatar_anchor_paths),
            "anchor_images_loaded": len(anchor_images),
            "has_avatar_anchor": has_avatar_anchor,
            "use_realistic_natural": use_realistic_natural,
        },
    )

    # 2) Generación base
    engine = "flux"

    if use_realistic_natural:
    print("[IsabelaOS] Using Realistic Vision for NATURAL + AVATAR")
    pipe = get_realistic_vision()
    engine = "realistic_vision"

    rv_steps = int(input_data.get("natural_rv_steps", 24))
    rv_guidance = float(input_data.get("natural_rv_guidance", 5.5))

    rv_prompt = (
        effective_prompt
        + ", single person, one woman only, solo portrait, one face only, "
          "head and shoulders, centered subject, symmetrical composition, "
          "looking at camera, realistic portrait"
    )

    rv_negative = (
        negative_prompt
        + ", multiple people, two people, group photo, duplicate face, extra face, "
          "extra head, extra body, merged body, fused body, twins, collage, stacked bodies, "
          "double head, cloned face, duplicated person, cropped second person"
    )

    print(
        "[IsabelaOS] RV prompt control:",
        {
            "rv_steps": rv_steps,
            "rv_guidance": rv_guidance,
            "rv_prompt": rv_prompt,
            "rv_negative": rv_negative,
        },
    )

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=DTYPE_SD15):
                image = pipe(
                    prompt=rv_prompt,
                    negative_prompt=rv_negative,
                    num_inference_steps=rv_steps,
                    guidance_scale=rv_guidance,
                    width=width,
                    height=height,
                ).images[0]
        else:
            image = pipe(
                prompt=rv_prompt,
                negative_prompt=rv_negative,
                num_inference_steps=rv_steps,
                guidance_scale=rv_guidance,
                width=width,
                height=height,
            ).images[0]
            else:
                image = pipe(
                    prompt=effective_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=rv_steps,
                    guidance_scale=rv_guidance,
                    width=width,
                    height=height,
                ).images[0]
    else:
        print("[IsabelaOS] Using FLUX pipeline")
        pipe = get_flux()
        engine = "flux"

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

    print("[IsabelaOS] txt2img base generation finished ✅")

    # 3) Identity lock SOLO si hay avatar + anchors
    identity_warning = None
    if has_avatar_anchor:
        print("[IsabelaOS] Applying identity lock...")
        image, identity_warning = _apply_identity_lock(image, anchor_images)
        if identity_warning:
            print("[IsabelaOS] Identity lock warning:", identity_warning)
        else:
            print("[IsabelaOS] Identity lock applied ✅")

    # 4) Skin natural refine
    # Solo se usa el refine viejo cuando skin_mode=natural PERO no entró al flujo Realistic Vision + avatar.
    natural_skin_warning = None
    if skin_mode == "natural" and not use_realistic_natural and not has_avatar_anchor:
        print("[IsabelaOS] Applying natural skin refine (no avatar fallback)...")
        image, natural_skin_warning = _refine_natural_skin(image, input_data)
        if natural_skin_warning:
            print("[IsabelaOS] Natural skin warning:", natural_skin_warning)
        else:
            print("[IsabelaOS] Natural skin refine applied ✅")

    enc = encode_image_jpg(image)
    return {
        **enc,
        "mode": "txt2img_flux",
        "engine": engine,
        "identity_warning": identity_warning,
        "natural_skin_warning": natural_skin_warning,
        "avatar": {
            "id": avatar_id,
            "name": avatar_name,
            "anchor_urls_count": len(avatar_anchor_urls),
            "anchor_paths_count": len(avatar_anchor_paths),
            "anchor_images_loaded": len(anchor_images),
            "identity_lock_applied": identity_warning is None and bool(anchor_images),
        },
        "params": {
            "steps": steps,
            "size": [width, height],
            "used_effective_prompt": bool(effective_prompt),
            "skin_mode": skin_mode,
            "use_realistic_natural": use_realistic_natural,
        },
    }


def handle_compose_scene(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    MONTAJE IA LOCAL
    Usa el fondo EXACTO subido por el usuario.
    No regenera el escenario.
    No usa Flux.
    No cambia la identidad del sujeto.
    """
    print("[IsabelaOS] handle_compose_scene() entered")

    from rembg import remove as rembg_remove

    fg_b64 = input_data.get("fg_image_b64") or input_data.get("person_image")
    bg_b64 = input_data.get("bg_image_b64") or input_data.get("background_image")

    if not fg_b64 or not bg_b64:
        return {"error": "MISSING_FG_OR_BG"}

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

    fg_pil = decode_image(fg_b64).convert("RGBA")
    bg_pil = decode_image(bg_b64).convert("RGB")

    bg_bgr = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
    bg_h, bg_w = bg_bgr.shape[:2]

    fg_rgba = rembg_remove(np.array(fg_pil)).astype(np.uint8)

    if fg_rgba.ndim != 3 or fg_rgba.shape[2] == 3:
        alpha = np.ones((fg_rgba.shape[0], fg_rgba.shape[1]), dtype=np.uint8) * 255
        fg_rgba = np.dstack([fg_rgba, alpha])

    fg_h, fg_w = fg_rgba.shape[:2]
    new_w = max(8, int(bg_w * scale))
    ratio = new_w / max(1, fg_w)
    new_h = max(8, int(fg_h * ratio))

    fg_rgba = cv2.resize(fg_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    cx = int(bg_w * x)
    cy = int(bg_h * y)

    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

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

    fg_a = _feather_alpha(fg_a, feather)
    mask255 = fg_a.copy()

    bg_roi = bg_bgr[by1:by2, bx1:bx2].copy()

    if color_match:
        fg_rgb = _match_color_simple(fg_rgb, bg_roi, mask255)

    if blend_mode == "alpha":
        alpha_f = (fg_a.astype(np.float32) / 255.0)[..., None]
        comp = (fg_rgb.astype(np.float32) * alpha_f) + (bg_roi.astype(np.float32) * (1.0 - alpha_f))
        bg_bgr[by1:by2, bx1:bx2] = np.clip(comp, 0, 255).astype(np.uint8)
    else:
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
        print("[IsabelaOS] handler invoked")
        print("[IsabelaOS] raw event type =", type(event).__name__)

        input_data = event.get("input") or {}
        action = _safe_text(input_data.get("action", "")).lower()

        print("[IsabelaOS] action =", action or "(empty)")
        print("[IsabelaOS] input keys =", list(input_data.keys()))

        if action == "health":
            return {
                "message": "IsabelaOS worker online (FLUX / Realistic Vision + anchors + identity lock + natural skin + compose scene)"
            }

        if action in ["generate", "txt2img_flux", "generate_image", "txt2img"]:
            return handle_txt2img(input_data)

        if action == "compose_scene":
            return handle_compose_scene(input_data)

        return {"error": "UNKNOWN_ACTION", "action": action}

    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        print(traceback.format_exc())
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
