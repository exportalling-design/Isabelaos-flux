# Dockerfile — IsabelaOS Studio · RunPod Serverless
# FIX principal: remover PYTHONPATH de CodeFormer que causa conflicto con basicsr
# FIX secundario: CMD explícito para lanzar el handler directamente
# ignorando el post-start script de runpod/base que falla con RUNPOD_PROJECT_ID
FROM runpod/base:0.6.2-cuda12.1.0

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

# PyTorch con CUDA 12.1
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

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
# exit code 127 = python no encontrado en PATH
# runpod/base usa python3 — además crear symlink por si acaso
RUN ln -sf /usr/bin/python3 /usr/bin/python || true

CMD ["python3", "-u", "/workspace/rp_handler.py"]
