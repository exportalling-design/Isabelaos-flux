FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Solo dependencias del sistema — sin pip install pesado
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Solo runpod — sin torch ni diffusers todavía
RUN pip install --no-cache-dir runpod==1.7.3

# Handler mínimo
COPY rp_handler.py /workspace/rp_handler.py

CMD ["python3", "-u", "rp_handler.py"]