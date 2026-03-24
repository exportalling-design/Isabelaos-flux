import os
import io
import base64
import traceback
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image
import torch
import runpod

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ----------------------------
# Cache paths (RunPod Volume)
# ----------------------------
BASE_VOLUME = "/runpod/volumes/isabelaos"
UPSCALE_MODELS_DIR = f"{BASE_VOLUME}/upscale_models"
os.makedirs(UPSCALE_MODELS_DIR, exist_ok=True)

REALESRGAN_MODEL_PATH = f"{UPSCALE_MODELS_DIR}/RealESRGAN_x4plus.pth"
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
realesrgan_upsampler = None

print("[IsabelaOS Upscale] Worker booting...")
print("[IsabelaOS Upscale] DEVICE =", DEVICE)
print("[IsabelaOS Upscale] BASE_VOLUME =", BASE_VOLUME)


def _safe_text(s: Any, max_len: int = 1200) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "").strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _safe_int(v, d=0):
    try:
        return int(v)
    except Exception:
        return d


def _ensure_file_from_url(url: str, local_path: str) -> str:
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = local_path + ".tmp"

    print(f"[IsabelaOS Upscale] Downloading model from: {url}")
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp_path, local_path)
    print(f"[IsabelaOS Upscale] Model cached at: {local_path}")
    return local_path


def _strip_data_url_prefix(b64_str: str) -> str:
    if not b64_str:
        return b64_str
    if "," in b64_str and b64_str.startswith("data:image"):
        return b64_str.split(",", 1)[1]
    return b64_str


def decode_image(b64_str: str) -> Image.Image:
    try:
        clean_b64 = _strip_data_url_prefix(b64_str)
        raw = base64.b64decode(clean_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"INVALID_IMAGE_B64: {str(e)}")


def encode_image_jpg(img: Image.Image, quality: int = 92) -> Dict[str, str]:
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64

    return {
        "ok": True,
        "image_b64": b64,
        "image_data_url": data_url,
        "mime": "image/jpeg",
        "result_b64": b64,
        "resultBase64": b64,
        "image": b64,
        "image_base64": b64,
        "data_url": data_url,
    }


def get_realesrgan_upsampler():
    global realesrgan_upsampler

    if realesrgan_upsampler is not None:
        return realesrgan_upsampler

    print("[IsabelaOS Upscale] Loading RealESRGAN upsampler...")
    os.makedirs(UPSCALE_MODELS_DIR, exist_ok=True)

    if not os.path.exists(REALESRGAN_MODEL_PATH):
        _ensure_file_from_url(REALESRGAN_MODEL_URL, REALESRGAN_MODEL_PATH)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    realesrgan_upsampler = RealESRGANer(
        scale=4,
        model_path=REALESRGAN_MODEL_PATH,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE == "cuda"),
        gpu_id=0 if DEVICE == "cuda" else None,
    )

    print("[IsabelaOS Upscale] RealESRGAN ready ✅")
    return realesrgan_upsampler


def apply_upscale(img: Image.Image, outscale: int = 2) -> Image.Image:
    upsampler = get_realesrgan_upsampler()

    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    print(f"[IsabelaOS Upscale] Applying RealESRGAN upscale x{outscale}...")
    output, _ = upsampler.enhance(bgr, outscale=outscale)

    out_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out_rgb)

    print(f"[IsabelaOS Upscale] Upscale done ✅ size={out_pil.size}")
    return out_pil


def handle_upscale(input_data: Dict[str, Any]) -> Dict[str, Any]:
    image_b64 = input_data.get("image_b64")
    if not image_b64:
        return {
            "ok": False,
            "error": "MISSING_IMAGE_B64",
            "received_keys": list(input_data.keys())
        }

    outscale = _safe_int(input_data.get("outscale", 2), 2)
    if outscale not in [2, 3, 4]:
        outscale = 2

    img = decode_image(image_b64)
    out = apply_upscale(img, outscale=outscale)

    enc = encode_image_jpg(out)
    return {
        **enc,
        "mode": "upscale_realesrgan",
        "engine": "realesrgan",
        "params": {
            "outscale": outscale,
            "size": list(out.size),
            "device": DEVICE,
        },
    }


def _extract_input(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}

    if isinstance(event.get("input"), dict):
        return event["input"]

    # fallback por si pruebas body directo
    return event


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        print("[IsabelaOS Upscale] handler invoked")
        print("[IsabelaOS Upscale] raw event keys =", list(event.keys()) if isinstance(event, dict) else type(event))

        input_data = _extract_input(event)
        print("[IsabelaOS Upscale] input keys =", list(input_data.keys()))

        action = _safe_text(input_data.get("action", "")).lower()

        # si no viene action pero sí imagen, asumimos upscale
        if not action and input_data.get("image_b64"):
            action = "upscale"

        print("[IsabelaOS Upscale] action =", action or "(empty)")

        if action == "health":
            return {
                "ok": True,
                "message": "IsabelaOS Upscale worker online (RealESRGAN)",
                "device": DEVICE,
                "model_path": REALESRGAN_MODEL_PATH,
            }

        if action in ["upscale", "enhance", "realesrgan"]:
            return handle_upscale(input_data)

        return {
            "ok": False,
            "error": "UNKNOWN_ACTION",
            "action": action,
            "expected": ["health", "upscale", "enhance", "realesrgan"],
            "received_keys": list(input_data.keys()),
        }

    except Exception as e:
        print("[IsabelaOS Upscale ERROR]", repr(e))
        print(traceback.format_exc())
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()[:4000],
        }


runpod.serverless.start({"handler": handler})
