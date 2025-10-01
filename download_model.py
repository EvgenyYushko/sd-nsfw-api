# -*- coding: utf-8 -*-
# download_model.py
import torch
from diffusers import StableDiffusionPipeline
import os

# Получаем ID модели из переменной окружения или используем значение по умолчанию
MODEL_ID = os.getenv("MODEL_ID", "NSFW-API/NSFW_Wan_14b")

print(f"Скачиваем модель {MODEL_ID}...")

# Просто скачиваем модель и ее компоненты. Они сохранятся в кэше.
StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)

print("Модель успешно скачана в кэш.")