
# ═══════════════════════════════════════════════════════════════════════════
# IsabelaOS — Worker RunPod Serverless
# Base: CUDA 12.1.1 + cuDNN 8 + Ubuntu 22.04
# Stack: FLUX + Realistic Vision + InsightFace + CodeFormer
# ═══════════════════════════════════════════════════════════════════════════
 
# Force rebuild v4
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
 
# Evitar prompts interactivos durante instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive
 
WORKDIR /workspace
 
# ── Dependencias del sistema ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    gcc \
    g++ \
    cmake \
    # libcublas-11-8 provee libcublasLt.so.11 que onnxruntime-gpu necesita
    libcublas-11-8 \
 && rm -rf /var/lib/apt/lists/*
 
# ── Fix librerías CUDA para onnxruntime-gpu ─────────────────────────────────
# onnxruntime-gpu busca libcublasLt.so.11 y libcufft.so.10
# pero CUDA 12.1 instala versiones más nuevas (.so.12, .so.11)
# Los symlinks resuelven: "cannot open shared object file"
RUN ln -sf /usr/local/cuda/lib64/libcublasLt.so \
           /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || true \
 && ln -sf /usr/local/cuda/lib64/libcufft.so \
           /usr/lib/x86_64-linux-gnu/libcufft.so.10 || true
 
# ── Actualizar pip ──────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip setuptools wheel
 
# ── PyTorch con CUDA 12.1 ───────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
 
# ── Dependencias Python ─────────────────────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
 
# ── CodeFormer ──────────────────────────────────────────────────────────────
# Necesitamos el repo completo por dos razones:
#   1. basicsr/archs/codeformer_arch.py — arquitectura del modelo
#   2. facelib — detección facial para FaceRestoreHelper
#
# IMPORTANTE: "python3 setup.py develop" genera basicsr/version.py
# Sin ese paso, basicsr/__init__.py falla con:
#   ModuleNotFoundError: No module named 'basicsr.version'
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
 && cd /workspace/CodeFormer \
 && pip install --no-cache-dir -r requirements.txt \
 && python3 setup.py develop
 
# Agregar CodeFormer al path de Python para que los imports funcionen
ENV PYTHONPATH="/workspace/CodeFormer"
 
# ── Handler del worker ──────────────────────────────────────────────────────
COPY rp_handler.py /workspace/rp_handler.py
 
# Iniciar el worker serverless de RunPod
CMD ["python3", "-u", "rp_handler.py"]
 
