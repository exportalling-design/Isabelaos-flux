
# rp_handler.py – IsabelaOS Studio v3
# FIXES v3:
#   1. BASE_VOLUME lee ISE_VOLUME_MOUNT correctamente
#   2. CodeFormer usa basicsr de pip (no del repo clonado)
#   3. Realistic Vision con prompts forzados de piel imperfecta real
#   4. Resolucion RV reducida a 512x768 para evitar personas duplicadas
#   5. Sufijos de prompt cortos para no truncar CLIP (limite 77 tokens)
 
import os, io, base64, urllib.request, traceback, hashlib
from typing import Dict, Any, Optional, List
 
import cv2, torch, numpy as np
from PIL import Image
import runpod
 
# ── Volumen y cache ────────────────────────────────────────────────────────
BASE_VOLUME = os.environ.get("ISE_VOLUME_MOUNT", "/runpod/volumes/isabela-video")
 
os.environ.setdefault("HF_HOME",           f"{BASE_VOLUME}/huggingface")
os.environ.setdefault("HF_HUB_CACHE",      f"{BASE_VOLUME}/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{BASE_VOLUME}/huggingface/transformers")
os.environ.setdefault("DIFFUSERS_CACHE",    f"{BASE_VOLUME}/huggingface/diffusers")
os.environ.setdefault("TORCH_HOME",         f"{BASE_VOLUME}/torch")
 
ANCHOR_CACHE_DIR   = f"{BASE_VOLUME}/avatar_anchors"
FACE_MODELS_DIR    = f"{BASE_VOLUME}/face_models"
CODEFORMER_WEIGHTS = f"{BASE_VOLUME}/codeformer/codeformer.pth"
 
for p in [
    os.environ["HF_HOME"], os.environ["HF_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"], os.environ["DIFFUSERS_CACHE"],
    os.environ["TORCH_HOME"], ANCHOR_CACHE_DIR, FACE_MODELS_DIR,
    os.path.dirname(CODEFORMER_WEIGHTS),
]:
    os.makedirs(p, exist_ok=True)
 
from diffusers import FluxPipeline, UniPCMultistepScheduler, StableDiffusionPipeline, AutoencoderKL
 
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_FLUX = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_SD15 = torch.float16 if DEVICE == "cuda" else torch.float32
 
FLUX_MODEL_ID      = os.environ.get("ISE_FLUX_MODEL_ID",  "black-forest-labs/FLUX.1-schnell")
REALISTIC_MODEL_ID = os.environ.get("REALISTIC_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE")
 
INSWAPPER_PATH = f"{FACE_MODELS_DIR}/inswapper_128.onnx"
INSWAPPER_URL  = os.environ.get(
    "INSWAPPER_MODEL_URL",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
)
 
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
 
flux_pipe = realistic_pipe = face_analyser = face_swapper = None
_cf_net   = _cf_helper = None
 
print("[IsabelaOS] Worker booting... DEVICE =", DEVICE, "| BASE_VOLUME =", BASE_VOLUME)
 
 
# ══════════════════════════════════════════════════════════════════════════
# PROMPTS FORZADOS PARA PIEL NATURAL IMPERFECTA
# Estos se inyectan automaticamente cuando el usuario elige skin_mode=natural
# Son cortos para no superar el limite de 77 tokens de CLIP
# ══════════════════════════════════════════════════════════════════════════
 
RV_SKIN_SUFFIX = (
    ", skin pores, freckles, moles, stretch marks, skin rolls, "
    "cellulite, acne scars, unretouched skin, one person only"
)
 
RV_SKIN_NEGATIVE = (
    ", smooth skin, perfect skin, airbrushed, plastic skin, "
    "beauty filter, flawless, two people, duplicate person, "
    "multiple faces, extra head, cloned face"
)
 
 
# ══════════════════════════════════════════════════════════════════════════
# PIPELINES
# ══════════════════════════════════════════════════════════════════════════
 
def get_flux():
    global flux_pipe
    if flux_pipe:
        return flux_pipe
    print("[IsabelaOS] Loading FLUX...")
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID, torch_dtype=DTYPE_FLUX,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    if DEVICE == "cuda":
        flux_pipe = flux_pipe.to("cuda")
    print("[IsabelaOS] FLUX ready ✅")
    return flux_pipe
 
 
def get_realistic_vision():
    global realistic_pipe
    if realistic_pipe:
        return realistic_pipe
    print("[IsabelaOS] Loading Realistic Vision...")
    realistic_pipe = StableDiffusionPipeline.from_pretrained(
        REALISTIC_MODEL_ID, torch_dtype=DTYPE_SD15,
        cache_dir=os.environ["HF_HUB_CACHE"],
        safety_checker=None, requires_safety_checker=False,
    )
    try:
        realistic_pipe.scheduler = UniPCMultistepScheduler.from_config(
            realistic_pipe.scheduler.config)
    except Exception:
        pass
    realistic_pipe.safety_checker = None
    if DEVICE == "cuda":
        realistic_pipe = realistic_pipe.to("cuda")
        try:
            realistic_pipe.enable_attention_slicing()
        except Exception:
            pass
    print("[IsabelaOS] Realistic Vision ready ✅")
    return realistic_pipe
 
 
# ══════════════════════════════════════════════════════════════════════════
# CODEFORMER
# Usa basicsr instalado por pip (no el del repo clonado de CodeFormer)
# Esto evita el error: ModuleNotFoundError: No module named 'basicsr.version'
# ══════════════════════════════════════════════════════════════════════════
 
def get_codeformer():
    global _cf_net, _cf_helper
    if _cf_net:
        return _cf_net, _cf_helper
    print("[IsabelaOS] Loading CodeFormer...")
 
    # Importar desde basicsr de pip, no del repo clonado
    import sys
    cf_repo = "/workspace/CodeFormer"
    if cf_repo in sys.path:
        sys.path.remove(cf_repo)
 
    from basicsr.archs.codeformer_arch import CodeFormer as CF
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
 
    if not os.path.exists(CODEFORMER_WEIGHTS):
        _ensure_file_from_url(
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            CODEFORMER_WEIGHTS,
        )
 
    net = CF(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(DEVICE)
 
    ck = torch.load(CODEFORMER_WEIGHTS, map_location=DEVICE)
    net.load_state_dict(ck["params_ema"])
    net.eval()
    _cf_net = net
 
    _cf_helper = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png",
        use_parse=True, device=DEVICE,
    )
    print("[IsabelaOS] CodeFormer ready ✅")
    return _cf_net, _cf_helper
 
 
def run_codeformer(img_pil: Image.Image, fidelity_weight: float = 0.90) -> Image.Image:
    # fidelity_weight: 0.90 = standard | 0.85 = natural
    try:
        from basicsr.utils import img2tensor, tensor2img
        net, helper = get_codeformer()
 
        img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        helper.clean_all()
        helper.read_image(img_bgr)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        helper.align_warp_face()
 
        if not helper.cropped_faces:
            print("[IsabelaOS] CodeFormer: sin caras detectadas, devolviendo original")
            return img_pil
 
        for cropped in helper.cropped_faces:
            t = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = net(t, w=fidelity_weight, adain=True)[0]
            restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
            helper.add_restored_face(restored, cropped)
 
        helper.get_inverse_affine(None)
        result = helper.paste_faces_to_input_image(
            upsample_img=None, draw_box=False, face_upsampler=None)
        print(f"[IsabelaOS] CodeFormer applied ✅ fidelity={fidelity_weight}")
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
 
    except Exception as e:
        print("[IsabelaOS] CodeFormer failed:", repr(e))
        print(traceback.format_exc())
        return img_pil
 
 
# ══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════
 
def encode_image_jpg(img: Image.Image, quality: int = 92) -> Dict[str, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    url = "data:image/jpeg;base64," + b64
    return {
        "image_b64": b64, "image_data_url": url, "mime": "image/jpeg",
        "result_b64": b64, "resultBase64": b64, "image": b64,
        "image_base64": b64, "data_url": url,
    }
 
 
def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
 
 
def is_flat(img: Image.Image) -> bool:
    try:
        return np.array(img.convert("RGB"), dtype=np.uint8).std() < 2.0
    except Exception:
        return False
 
 
def _safe_text(s, max_len=1200):
    return ("" if s is None else str(s)).replace("\x00", "").strip()[:max_len]
 
def _safe_float(v, d=0.0):
    try: return float(v)
    except Exception: return d
 
def _safe_int(v, d=0):
    try: return int(v)
    except Exception: return d
 
def _clamp(x, a, b): return max(a, min(b, x))
 
def _safe_list(v):
    if not isinstance(v, list): return []
    return [_safe_text(i, 4000) for i in v if _safe_text(i, 4000)]
 
def pil_to_bgr(img): return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
def bgr_to_pil(arr): return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
 
 
# ══════════════════════════════════════════════════════════════════════════
# ANCHORS
# ══════════════════════════════════════════════════════════════════════════
 
def _hash_url(url): return hashlib.sha1(url.encode()).hexdigest()
 
def _guess_ext(url):
    for ext in [".png", ".webp", ".jpeg", ".jpg"]:
        if ext in url.lower(): return ext.lstrip(".")
    return "jpg"
 
def _ensure_file_from_url(url: str, local_path: str) -> str:
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp = local_path + ".tmp"
    print(f"[IsabelaOS] Descargando: {url[:80]}...")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)
    os.replace(tmp, local_path)
    print(f"[IsabelaOS] Guardado: {local_path}")
    return local_path
 
def _load_anchor_images(urls: List[str], avatar_id: Optional[str]) -> List[Image.Image]:
    images = []
    for i, url in enumerate(urls[:3]):
        try:
            path = os.path.join(
                ANCHOR_CACHE_DIR,
                f"{avatar_id or 'default'}_{i+1}_{_hash_url(url)}.{_guess_ext(url)}"
            )
            _ensure_file_from_url(url, path)
            images.append(Image.open(path).convert("RGB"))
        except Exception as e:
            print("[IsabelaOS] WARNING anchor:", repr(e))
    return images
 
 
# ══════════════════════════════════════════════════════════════════════════
# INSIGHTFACE
# ══════════════════════════════════════════════════════════════════════════
 
def _ort_providers():
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers() and DEVICE == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]
 
def get_face_analyser():
    global face_analyser
    if face_analyser: return face_analyser
    print("[IsabelaOS] Loading FaceAnalysis...")
    from insightface.app import FaceAnalysis
    face_analyser = FaceAnalysis(
        name="buffalo_l", root=FACE_MODELS_DIR, providers=_ort_providers())
    face_analyser.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
    print("[IsabelaOS] FaceAnalysis ready ✅")
    return face_analyser
 
def get_face_swapper():
    global face_swapper
    if face_swapper: return face_swapper
    print("[IsabelaOS] Loading face swapper...")
    from insightface.model_zoo import get_model
    if not os.path.exists(INSWAPPER_PATH):
        _ensure_file_from_url(INSWAPPER_URL, INSWAPPER_PATH)
    face_swapper = get_model(INSWAPPER_PATH, providers=_ort_providers())
    print("[IsabelaOS] Face swapper ready ✅")
    return face_swapper
 
def _pick_largest(faces):
    if not faces: return None
    return sorted(
        faces,
        key=lambda f: max(0, f.bbox[2]-f.bbox[0]) * max(0, f.bbox[3]-f.bbox[1]),
        reverse=True
    )[0]
 
def _apply_identity_lock(image: Image.Image, anchors: List[Image.Image]):
    if not anchors: return image, "IDENTITY_LOCK_SKIPPED_NO_ANCHORS"
    try:
        analyser = get_face_analyser()
        swapper  = get_face_swapper()
        src_face = None
        for i, a in enumerate(anchors):
            src_face = _pick_largest(analyser.get(pil_to_bgr(a)))
            if src_face:
                print(f"[IsabelaOS] Cara fuente encontrada en anchor {i+1}")
                break
        if not src_face: return image, "IDENTITY_LOCK_NO_FACE_IN_ANCHORS"
        tgt_bgr  = pil_to_bgr(image)
        tgt_face = _pick_largest(analyser.get(tgt_bgr))
        if not tgt_face: return image, "IDENTITY_LOCK_NO_FACE_IN_GENERATION"
        swapped  = swapper.get(tgt_bgr, tgt_face, src_face, paste_back=True)
        print("[IsabelaOS] Identity lock applied ✅")
        return bgr_to_pil(swapped), None
    except Exception as e:
        print("[IsabelaOS] Identity lock failed:", repr(e))
        return image, f"IDENTITY_LOCK_FAILED: {e}"
 
 
# ══════════════════════════════════════════════════════════════════════════
# MONTAJE IA (compose_scene)
# ══════════════════════════════════════════════════════════════════════════
 
def _feather_alpha(a, px):
    if px <= 0: return a
    k = max(3, px * 2 + 1)
    return cv2.GaussianBlur(a, (k, k), 0)
 
def _match_color(fg, bg, mask):
    m = mask > 0
    if m.sum() < 50: return fg
    out = fg.astype(np.float32)
    for c in range(3):
        fv   = fg[..., c][m]
        ring = cv2.dilate(mask, np.ones((31, 31), np.uint8)) > 0
        bv   = bg[..., c][ring]
        if bv.size < 50: continue
        out[..., c] = (out[..., c] - fv.mean()) * (bv.std() + 1e-6) / (fv.std() + 1e-6) + bv.mean()
    return np.clip(out, 0, 255).astype(np.uint8)
 
def _contact_shadow(bg, mask, box, opacity=0.18):
    x1, y1, x2, y2 = box
    out = bg.copy()
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0: return out
    sh  = (mask > 10).astype(np.uint8) * 255
    sh  = cv2.resize(sh, (w, h))
    ch  = max(8, int(h * 0.18))
    sm  = cv2.GaussianBlur(cv2.resize(sh, (w, ch)), (0, 0), sigmaX=9, sigmaY=5)
    canvas = np.zeros((bg.shape[0], bg.shape[1]), np.float32)
    sy1 = min(bg.shape[0] - 1, max(0, y2 - ch // 2))
    sy2 = min(bg.shape[0], sy1 + ch)
    sx1, sx2 = max(0, x1), min(bg.shape[1], x2)
    if sy2 > sy1 and sx2 > sx1:
        canvas[sy1:sy2, sx1:sx2] = sm[:sy2-sy1, :sx2-sx1].astype(np.float32) / 255
    canvas = np.clip(cv2.GaussianBlur(canvas, (0, 0), sigmaX=12, sigmaY=8) * opacity, 0, 1)
    for c in range(3):
        out[..., c] = out[..., c].astype(np.float32) * (1 - canvas)
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
# ══════════════════════════════════════════════════════════════════════════
# ACTION: TXT2IMG
# ══════════════════════════════════════════════════════════════════════════
 
def handle_txt2img(inp: Dict[str, Any]) -> Dict[str, Any]:
    print("[IsabelaOS] handle_txt2img()")
 
    prompt    = _safe_text(inp.get("prompt", ""))
    eff       = _safe_text(inp.get("effective_prompt", "")) or prompt
    neg       = _safe_text(inp.get("negative_prompt", ""))
    skin_mode = _safe_text(inp.get("skin_mode", "standard")).lower() or "standard"
 
    for p in ["across frames", "frame skipping", "motion artifacts", "gentle blinking", "temporal wobble"]:
        eff = eff.replace(p, "")
 
    steps  = _safe_int(inp.get("steps", 4))
    width  = _safe_int(inp.get("width", 1024))
    height = _safe_int(inp.get("height", 1024))
 
    avatar_id   = _safe_text(inp.get("avatar_id", "")) or None
    avatar_name = _safe_text(inp.get("avatar_name", "")) or None
    anchor_urls = _safe_list(inp.get("avatar_anchor_urls"))
 
    cf_fidelity = 0.90 if skin_mode == "standard" else 0.85
 
    anchors = []
    if anchor_urls:
        try:
            anchors = _load_anchor_images(anchor_urls, avatar_id)
        except Exception as e:
            print("[IsabelaOS] WARNING anchors:", repr(e))
 
    has_anchor     = bool(avatar_id and anchors)
    use_rv_natural = bool(has_anchor and skin_mode == "natural")
 
    # Resolucion reducida para RV: evita personas duplicadas y cabezas extras
    # 512x768 es el sweet spot para Realistic Vision V5
    if use_rv_natural:
        width, height = 512, 768
 
    print("[pipeline]", {
        "skin_mode": skin_mode, "cf_fidelity": cf_fidelity,
        "size": [width, height], "anchors": len(anchors),
        "has_anchor": has_anchor, "use_rv_natural": use_rv_natural,
    })
 
    engine = "flux"
 
    if use_rv_natural:
        pipe   = get_realistic_vision()
        engine = "realistic_vision"
 
        rv_steps    = _safe_int(inp.get("natural_rv_steps", 28))
        rv_guidance = _safe_float(inp.get("natural_rv_guidance", 7.0))
 
        # Sufijo corto para no truncar CLIP (limite 77 tokens)
        # Fuerza piel imperfecta real: poros, manchas, estrias, rollos
        rv_prompt = eff[:200] + RV_SKIN_SUFFIX
        rv_neg    = neg[:100] + RV_SKIN_NEGATIVE
 
        print(f"[RV] prompt ({len(rv_prompt)} chars): {rv_prompt[:80]}...")
        print(f"[RV] steps={rv_steps} guidance={rv_guidance} size={width}x{height}")
 
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=DTYPE_SD15) if DEVICE == "cuda" else torch.no_grad()
            with ctx:
                image = pipe(
                    prompt=rv_prompt, negative_prompt=rv_neg,
                    num_inference_steps=rv_steps, guidance_scale=rv_guidance,
                    width=width, height=height,
                ).images[0]
    else:
        pipe = get_flux()
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=DTYPE_FLUX) if DEVICE == "cuda" else torch.no_grad()
            with ctx:
                image = pipe(
                    prompt=eff, num_inference_steps=steps,
                    width=width, height=height,
                ).images[0]
 
    print("[IsabelaOS] Generacion base completada ✅")
 
    id_warn = None
    if has_anchor:
        image, id_warn = _apply_identity_lock(image, anchors)
        if id_warn:
            print("[IsabelaOS] id_warn:", id_warn)
 
    cf_applied = False
    cf_warn    = None
 
    if has_anchor or skin_mode == "natural":
        print(f"[IsabelaOS] CodeFormer fidelity={cf_fidelity}...")
        try:
            out_cf = run_codeformer(image, fidelity_weight=cf_fidelity)
            if not is_flat(out_cf):
                image      = out_cf
                cf_applied = True
            else:
                cf_warn = "CODEFORMER_FLAT_OUTPUT"
        except Exception as e:
            cf_warn = f"CODEFORMER_FAILED: {e}"
            print("[IsabelaOS] CodeFormer failed:", repr(e))
 
    enc = encode_image_jpg(image)
    return {
        **enc,
        "mode": "txt2img", "engine": engine,
        "identity_warning":   id_warn,
        "codeformer_applied": cf_applied,
        "codeformer_warning": cf_warn,
        "avatar": {
            "id": avatar_id, "name": avatar_name,
            "anchors_loaded": len(anchors),
            "identity_lock_applied": id_warn is None and bool(anchors),
        },
        "params": {
            "steps": steps, "size": [width, height],
            "skin_mode": skin_mode, "cf_fidelity": cf_fidelity,
        },
    }
 
 
# ══════════════════════════════════════════════════════════════════════════
# ACTION: COMPOSE SCENE (Montaje IA)
# ══════════════════════════════════════════════════════════════════════════
 
def handle_compose_scene(inp: Dict[str, Any]) -> Dict[str, Any]:
    print("[IsabelaOS] handle_compose_scene()")
    from rembg import remove as rembg_remove
 
    fg_b64 = inp.get("fg_image_b64") or inp.get("person_image")
    bg_b64 = inp.get("bg_image_b64") or inp.get("background_image")
    if not fg_b64 or not bg_b64:
        return {"error": "MISSING_FG_OR_BG"}
 
    x       = _clamp(_safe_float(inp.get("x", 0.5)),      0.0, 1.0)
    y       = _clamp(_safe_float(inp.get("y", 0.72)),     0.0, 1.0)
    scale   = _clamp(_safe_float(inp.get("scale", 0.55)), 0.1, 2.0)
    feather = _clamp(_safe_int(inp.get("feather", 12)),   0, 40)
    mode    = _safe_text(inp.get("mode", "seamless")).lower() or "seamless"
    cmatch  = bool(inp.get("color_match", True))
    shadow  = bool(inp.get("shadow", True))
 
    fg_pil  = decode_image(fg_b64).convert("RGBA")
    bg_pil  = decode_image(bg_b64).convert("RGB")
    bg_bgr  = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
    bg_h, bg_w = bg_bgr.shape[:2]
 
    fg_rgba = rembg_remove(np.array(fg_pil)).astype(np.uint8)
    if fg_rgba.ndim != 3 or fg_rgba.shape[2] == 3:
        fg_rgba = np.dstack([fg_rgba, np.ones(fg_rgba.shape[:2], np.uint8) * 255])
 
    fh, fw = fg_rgba.shape[:2]
    nw = max(8, int(bg_w * scale))
    nh = max(8, int(fh * (nw / max(1, fw))))
    fg_rgba = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
 
    cx, cy = int(bg_w * x), int(bg_h * y)
    x1, y1 = cx - nw // 2, cy - nh // 2
    x2, y2 = x1 + nw, y1 + nh
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(bg_w, x2), min(bg_h, y2)
    if bx2 <= bx1 or by2 <= by1:
        return {"error": "PLACEMENT_OUT_OF_BOUNDS"}
 
    crop   = fg_rgba[by1-y1:by1-y1+(by2-by1), bx1-x1:bx1-x1+(bx2-bx1)]
    fg_rgb = crop[..., :3]
    fg_a   = _feather_alpha(crop[..., 3], feather)
    bg_roi = bg_bgr[by1:by2, bx1:bx2].copy()
 
    if cmatch:
        fg_rgb = _match_color(fg_rgb, bg_roi, fg_a)
 
    if mode == "alpha":
        af = (fg_a.astype(np.float32) / 255)[..., None]
        bg_bgr[by1:by2, bx1:bx2] = np.clip(
            fg_rgb.astype(np.float32) * af + bg_roi.astype(np.float32) * (1 - af),
            0, 255).astype(np.uint8)
    else:
        center    = (bx1 + (bx2-bx1)//2, by1 + (by2-by1)//2)
        full_mask = np.zeros((bg_h, bg_w), np.uint8)
        full_mask[by1:by2, bx1:bx2] = (fg_a > 10).astype(np.uint8) * 255
        src = np.zeros_like(bg_bgr)
        src[by1:by2, bx1:bx2] = fg_rgb
        try:
            bg_bgr = cv2.seamlessClone(src, bg_bgr, full_mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print("[compose_scene] seamlessClone fallback:", repr(e))
            af = (fg_a.astype(np.float32) / 255)[..., None]
            bg_bgr[by1:by2, bx1:bx2] = np.clip(
                fg_rgb.astype(np.float32) * af + bg_roi.astype(np.float32) * (1 - af),
                0, 255).astype(np.uint8)
 
    if shadow:
        bg_bgr = _contact_shadow(bg_bgr, fg_a, (bx1, by1, bx2, by2))
 
    enc = encode_image_jpg(Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)))
    return {
        **enc, "mode": "compose_scene", "engine": "local_compositor",
        "params": {"x": x, "y": y, "scale": scale, "feather": feather, "mode": mode},
    }
 
 
# ══════════════════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
 
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        print("[IsabelaOS] handler invoked")
        inp    = event.get("input") or {}
        action = _safe_text(inp.get("action", "")).lower()
        print("[IsabelaOS] action =", action, "| keys =", list(inp.keys()))
 
        if action == "health":
            return {"status": "ok", "volume": BASE_VOLUME, "device": DEVICE}
 
        if action in ["generate", "txt2img_flux", "generate_image", "txt2img"]:
            return handle_txt2img(inp)
 
        if action == "compose_scene":
            return handle_compose_scene(inp)
 
        return {"error": "UNKNOWN_ACTION", "action": action}
 
    except Exception as e:
        print("[IsabelaOS ERROR]", repr(e))
        print(traceback.format_exc())
        return {"error": str(e)}
 
 
runpod.serverless.start({"handler": handler})
 
