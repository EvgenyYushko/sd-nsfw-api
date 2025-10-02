# app.py
import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import os
from contextlib import asynccontextmanager
import traceback # <-- Добавляем импорт для подробных логов

# Создаем словарь для хранения нашей модели.
# Это лучший способ управлять состоянием в FastAPI.
ml_models = {}

# Эта функция будет выполняться при старте и остановке приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Код, который выполняется при старте ---
    print("Lifespan: Приложение запускается. Начинаем загрузку модели...")
    try:
        MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
        
        # Загружаем модель из локальных файлов, которые мы "запекли" в образ
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,      # <-- Хорошая практика, указываем формат
            local_files_only=True      # <-- УБЕДИТЕСЬ, ЧТО ЭТА СТРОКА ЕСТЬ!
        ).to("cuda")
        
        ml_models["sd_pipeline"] = pipe
        print("Lifespan: Модель успешно загружена и готова к работе!")

    except Exception as e:
        # !!! ЭТО САМАЯ ВАЖНАЯ ЧАСТЬ !!!
        # Если при загрузке модели произойдет ЛЮБАЯ ошибка, мы ее поймаем и распечатаем
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ !!!")
        print(f"!!! Тип ошибки: {type(e)}")
        print(f"!!! Текст ошибки: {e}")
        print("!!! Полный traceback ошибки:")
        traceback.print_exc() # Распечатываем полный стектрейс ошибки в логи
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    yield
    
    # --- Код, который выполняется при остановке ---
    print("Lifespan: Приложение останавливается. Очищаем ресурсы.")
    ml_models.clear()


# Создаем приложение FastAPI и передаем ему нашу функцию lifespan
app = FastAPI(lifespan=lifespan)

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    width: int = 512
    height: int = 512

@app.post("/generate")
def generate(body: RequestBody):
    # Проверяем, загрузилась ли модель
    if "sd_pipeline" not in ml_models:
        return {"error": "Модель не загружена из-за ошибки при старте. Проверьте логи."}

    # Берем модель из нашего хранилища
    pipe = ml_models["sd_pipeline"]
    
    image = pipe(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        num_inference_steps=body.steps,
        height=body.height,
        width=body.width
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_base64": img_str}