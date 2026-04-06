FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
COPY rp_handler.py /workspace/rp_handler.py
COPY start.sh /workspace/start.sh

RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]
