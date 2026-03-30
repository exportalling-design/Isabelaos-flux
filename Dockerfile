# Dockerfile — IsabelaOS FLUX Worker
# Base oficial de RunPod — apt-get funciona correctamente
FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# FFmpeg y dependencias del sistema — funciona con runpod/base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    gcc \
    g++ \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Links de CUDA para compatibilidad
RUN ln -sf /usr/local/cuda/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || true
RUN ln -sf /usr/local/cuda/lib64/libcufft.so /usr/lib/x86_64-linux-gnu/libcufft.so.10 || true

# Actualizar pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch con CUDA 12.1
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias del proyecto
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar CodeFormer
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
    && cd /workspace/CodeFormer \
    && pip install --no-cache-dir -r requirements.txt \
    && python3 basicsr/setup.py develop

ENV PYTHONPATH="/workspace/CodeFormer"

# Copiar handler
COPY rp_handler.py /workspace/rp_handler.py

CMD ["python3", "-u", "rp_handler.py"]
