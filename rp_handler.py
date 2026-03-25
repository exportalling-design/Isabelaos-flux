def handle_txt2img(input_data: Dict[str, Any]) -> Dict[str, Any]:
    print("[IsabelaOS] handle_txt2img() entered")

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

    # LoRA temporalmente desactivado para probar solo anchors + face swap
    avatar_lora_path = None

    avatar_anchor_urls = _safe_list(input_data.get("avatar_anchor_urls"))
    avatar_anchor_paths = _safe_list(input_data.get("avatar_anchor_paths"))

    # 1) Cargar LoRA si hay avatar
    lora_info = _ensure_flux_avatar_lora(pipe, avatar_lora_path, avatar_id)

    # 2) Cargar anchors si existen
    anchor_images = []
    if avatar_anchor_urls:
        try:
            anchor_images = _load_anchor_images(avatar_anchor_urls, avatar_id)
        except Exception as e:
            print("[IsabelaOS] WARNING: anchor loading failed:", repr(e))

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
            "avatar_anchor_urls_count": len(avatar_anchor_urls),
            "avatar_anchor_paths_count": len(avatar_anchor_paths),
            "anchor_images_loaded": len(anchor_images),
            "used_lora": lora_info.get("used_lora"),
        },
    )

    # 3) Generación base FLUX
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

    # 4) Identity lock SOLO si hay anchors
    identity_warning = None
    if anchor_images:
        print("[IsabelaOS] Applying identity lock...")
        image, identity_warning = _apply_identity_lock(image, anchor_images)
        if identity_warning:
            print("[IsabelaOS] Identity lock warning:", identity_warning)
        else:
            print("[IsabelaOS] Identity lock applied ✅")

    # 5) Upscale SOLO si hay anchors
    upscale_warning = None
    if anchor_images:
        print("[IsabelaOS] Applying upscale after identity lock...")
        image, upscale_warning = _call_upscale_endpoint_if_needed(image, should_upscale=True)
        if upscale_warning:
            print("[IsabelaOS] Upscale warning:", upscale_warning)
        else:
            print("[IsabelaOS] Upscale applied after identity lock ✅")

    enc = encode_image_jpg(image)
    return {
        **enc,
        "mode": "txt2img_flux",
        "engine": "flux",
        "warning": lora_info.get("warning"),
        "identity_warning": identity_warning,
        "upscale_warning": upscale_warning,
        "avatar": {
            "id": avatar_id,
            "name": avatar_name,
            "trigger": avatar_trigger,
            "lora_path": avatar_lora_path,
            "used_lora": lora_info.get("used_lora", False),
            "anchor_urls_count": len(avatar_anchor_urls),
            "anchor_paths_count": len(avatar_anchor_paths),
            "anchor_images_loaded": len(anchor_images),
            "identity_lock_applied": identity_warning is None and bool(anchor_images),
            "upscale_applied": upscale_warning is None and bool(anchor_images),
        },
        "params": {
            "steps": steps,
            "size": [width, height],
            "used_effective_prompt": bool(effective_prompt),
        },
    }
