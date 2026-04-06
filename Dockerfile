# Dockerfile — IsabelaOS Studio · RunPod Serverless
# FIX principal: remover PYTHONPATH de CodeFormer que causa conflicto con basicsr
# FIX secundario: CMD explícito para lanzar el handler directamente
# ignorando el post-start script de runpod/base que falla con RUNPOD_PROJECT_ID
FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Dependencias del sistema — incluye python3-pip y opencv del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Symlink python → python3
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# RunPod SDK — usar python3 -m pip para asegurar el Python correcto
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir runpod==1.7.3

# PyTorch con CUDA 12.1
RUN python3 -m pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# CodeFormer — clonar e instalar dependencias
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
    && cd /workspace/CodeFormer \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# NO exportar PYTHONPATH con CodeFormer
# El rp_handler.py lo remueve del sys.path manualmente si está presente
# ENV PYTHONPATH="/workspace/CodeFormer"  ← REMOVIDO, esta era la causa del bug

# Handler principal
COPY rp_handler.py /workspace/rp_handler.py

CMD ["python3", "-u", "/workspace/rp_handler.py"]
