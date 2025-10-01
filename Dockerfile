# --- ЭТАП 1: "СБОРЩИК МОДЕЛИ" ---
FROM python:3.10-slim as builder

# Установка зависимостей для скачивания
# git и git-lfs нужны для скачивания некоторых моделей с HF
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors

# Указываем корневую папку для кэша huggingface
# Важно: это та папка, которую мы потом скопируем
ENV HF_HOME=/hf_cache

# Копируем и запускаем наш скрипт для скачивания
COPY download_model.py .
RUN python download_model.py

# --- ЭТАП 2: "ФИНАЛЬНЫЙ ОБРАЗ ПРИЛОЖЕНИЯ" ---
FROM python:3.10-slim

WORKDIR /app

# Устанавливаем только те зависимости, что нужны для работы API
# PyTorch и diffusers уже будут установлены, но лучше указать их явно
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow

# Копируем ВСЮ папку с кэшем модели из сборщика в финальный образ
# Это самый важный шаг!
COPY --from=builder /hf_cache /root/.cache/huggingface

# Копируем основной код приложения
COPY app.py .

# Команда для запуска веб-сервера. Используйте переменную PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]