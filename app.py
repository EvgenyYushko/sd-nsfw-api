import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

MODEL_ID = os.getenv("MODEL_ID", "NSFW-API/NSFW_Wan_14b")

print(f"?? Загружаем модель {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True  # <-- ВОТ ЭТО ИЗМЕНЕНИЕ
).to("cuda")

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    width: int = 512
    height: int = 512

@app.post("/generate")
def generate(body: RequestBody):
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
