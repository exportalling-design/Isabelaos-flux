# Force rebuild v3
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
 
# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git ffmpeg libsm6 libxext6 libgl1 \
    build-essential gcc g++ cmake \
    # Fix: libcublasLt.so.11 para onnxruntime-gpu con CUDA 12.1
    libcublas-11-8 \
 && rm -rf /var/lib/apt/lists/*
 
# Symlink para onnxruntime-gpu que busca libcublasLt.so.11
RUN ln -sf /usr/local/cuda/lib64/libcublasLt.so \
           /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || true
 
RUN python3 -m pip install --upgrade pip setuptools wheel
 
# ── PyTorch CUDA 12.1 ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
 
# ── Python deps ────────────────────────────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
 
# ── CodeFormer arch (necesita el repo para basicsr.archs.codeformer_arch) ──
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
 && cd /workspace/CodeFormer \
 && pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CodeFormer"
 
# ── Handler ────────────────────────────────────────────────────────────────
COPY rp_handler.py /workspace/rp_handler.py
 
CMD ["python3", "-u", "rp_handler.py"]
