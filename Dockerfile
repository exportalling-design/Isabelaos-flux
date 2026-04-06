FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir runpod==1.7.3 \
    && pip3 install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
       --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt \
    && git clone --depth=1 https://github.com/sczhou/CodeFormer /workspace/CodeFormer \
    && pip3 install --no-cache-dir -r /workspace/CodeFormer/requirements.txt

COPY rp_handler.py /workspace/rp_handler.py

CMD ["python3", "-u", "/workspace/rp_handler.py"]
