#!/bin/bash
# start.sh — IsabelaOS Studio RunPod Worker
# Instala dependencias en runtime para mantener el Dockerfile mínimo
# RunPod no aguanta Dockerfiles pesados en el builder

set -e

echo "[IsabelaOS] Iniciando setup en runtime..."

# ── Python y pip ──────────────────────────────────────────────
# runpod/base ya tiene python3, asegurar que pip apunta al correcto
which python3 && python3 --version || echo "python3 no encontrado"
which pip3 && pip3 --version || echo "pip3 no encontrado"

# Crear symlinks
ln -sf $(which python3) /usr/bin/python 2>/dev/null || true
ln -sf $(which pip3) /usr/bin/pip 2>/dev/null || true

# ── Upgrade pip ───────────────────────────────────────────────
python3 -m pip install --no-cache-dir --quiet --upgrade pip

# ── RunPod SDK ────────────────────────────────────────────────
echo "[IsabelaOS] Instalando RunPod SDK..."
python3 -m pip install --no-cache-dir --quiet runpod==1.7.3

# ── PyTorch con CUDA 12.1 ─────────────────────────────────────
echo "[IsabelaOS] Instalando PyTorch..."
python3 -m pip install --no-cache-dir --quiet \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Dependencias del proyecto ─────────────────────────────────
echo "[IsabelaOS] Instalando requirements..."
python3 -m pip install --no-cache-dir --quiet -r /workspace/requirements.txt

# ── CodeFormer ────────────────────────────────────────────────
# NO agregar al PYTHONPATH — el handler lo remueve de sys.path
if [ ! -d "/workspace/CodeFormer" ]; then
    echo "[IsabelaOS] Clonando CodeFormer..."
    git clone --quiet https://github.com/sczhou/CodeFormer /workspace/CodeFormer
    python3 -m pip install --no-cache-dir --quiet -r /workspace/CodeFormer/requirements.txt
else
    echo "[IsabelaOS] CodeFormer ya existe, saltando..."
fi

# ── Verificar cv2 ─────────────────────────────────────────────
echo "[IsabelaOS] Verificando imports críticos..."
python3 -c "import cv2; print('[OK] cv2', cv2.__version__)" || {
    echo "[WARN] cv2 no disponible, reinstalando..."
    python3 -m pip install --no-cache-dir opencv-python-headless==4.10.0.84
    python3 -c "import cv2; print('[OK] cv2 reinstalado', cv2.__version__)"
}
python3 -c "import torch; print('[OK] torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import numpy; print('[OK] numpy', numpy.__version__)"

echo "[IsabelaOS] Setup completo. Iniciando handler..."

# ── Lanzar el handler ─────────────────────────────────────────
exec python3 -u /workspace/rp_handler.py
