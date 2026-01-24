import os
import time
import requests
import json

# Toma los valores de variables de entorno
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
    raise RuntimeError("Faltan RUNPOD_API_KEY o RUNPOD_ENDPOINT_ID en las variables de entorno.")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
}

# Este input debe coincidir con lo que usa tu rp_handler
job_input = {
    "prompt": "a beautiful cinematic portrait, high detail, dramatic lighting",
    "negative_prompt": "low quality, blurry, deformed, text",
    "width": 512,
    "height": 512,
    "steps": 22
}

# 1) Lanzar job
print(f"Enviando a: {BASE_URL}/run\n")

resp = requests.post(
    f"{BASE_URL}/run",
    headers=headers,
    json={"input": job_input},
)

print("Status inicial:", resp.status_code)
print("Body inicial:", resp.text)

resp.raise_for_status()
job_id = resp.json()["id"]
print("Job ID:", job_id)

# 2) Polling hasta que termine
status_url = f"{BASE_URL}/status/{job_id}"

while True:
    time.sleep(5)
    s = requests.get(status_url, headers=headers)
    s.raise_for_status()
    data = s.json()
    print("Estado actual:", data.get("status"))

    if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
        print("Respuesta final:\n", json.dumps(data, indent=2))
        break

# Si terminó bien, mostrar la ruta de la imagen (ajustar a tu rp_handler)
if data.get("status") == "COMPLETED":
    output = data.get("output")
    print("OUTPUT:", output)
