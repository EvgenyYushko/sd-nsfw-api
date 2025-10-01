FROM python:3.10-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Установка PyTorch с CUDA 
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Diffusers + API
RUN pip install diffusers transformers accelerate safetensors fastapi uvicorn pillow

WORKDIR /app
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
