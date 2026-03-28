FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg libsm6 libxext6 libgl1 build-essential gcc g++ cmake libcublas-11-8 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/local/cuda/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || true
RUN ln -sf /usr/local/cuda/lib64/libcufft.so /usr/lib/x86_64-linux-gnu/libcufft.so.10 || true
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/sczhou/CodeFormer /workspace/CodeFormer && cd /workspace/CodeFormer && pip install --no-cache-dir -r requirements.txt && python3 setup.py develop
ENV PYTHONPATH="/workspace/CodeFormer"
COPY rp_handler.py /workspace/rp_handler.py
CMD ["python3", "-u", "rp_handler.py"]
