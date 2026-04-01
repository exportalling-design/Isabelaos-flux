FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod==1.7.3

RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CodeFormer se clona pero NO se compila durante el build
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
    && cd /workspace/CodeFormer \
    && pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/workspace/CodeFormer"

COPY rp_handler.py /workspace/rp_handler.py