FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

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
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch GPU (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.upscale.txt /workspace/requirements.upscale.txt
RUN pip install --no-cache-dir -r requirements.upscale.txt

COPY . /workspace

CMD ["python3", "-u", "rp_upscale.py"]
