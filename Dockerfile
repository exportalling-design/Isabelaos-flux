# Dockerfile — IsabelaOS Studio · RunPod Serverless
# FIX: usar runpod/pytorch como base en lugar de runpod/base
# runpod/base:0.6.2 tiene scripts de inicio para desarrollo local
# que fallan en serverless porque RUNPOD_PROJECT_ID no existe
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RunPod SDK
RUN pip install --no-cache-dir runpod==1.7.3

# PyTorch ya viene en la imagen base — solo instalar el resto
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CodeFormer — clonar e instalar dependencias
# IMPORTANTE: NO agregar /workspace/CodeFormer al PYTHONPATH
# porque su basicsr no tiene el módulo 'version' y rompe el import
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
    && cd /workspace/CodeFormer \
    && pip install --no-cache-dir -r requirements.txt

# NO exportar PYTHONPATH con CodeFormer
# El rp_handler.py lo remueve del sys.path manualmente si está presente
# ENV PYTHONPATH="/workspace/CodeFormer"  ← REMOVIDO, esta era la causa del bug

# Handler principal
COPY rp_handler.py /workspace/rp_handler.py

# Arrancar el handler directamente
CMD ["python", "-u", "/workspace/rp_handler.py"]
