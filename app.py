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
import traceback # <-- ��������� ������ ��� ��������� �����

# ������� ������� ��� �������� ����� ������.
# ��� ������ ������ ��������� ���������� � FastAPI.
ml_models = {}

# ��� ������� ����� ����������� ��� ������ � ��������� ����������
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- ���, ������� ����������� ��� ������ ---
    print("Lifespan: ���������� �����������. �������� �������� ������...")
    try:
        MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
        
        # ��������� ������ �� ��������� ������, ������� �� "�������" � �����
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,      # <-- ������� ��������, ��������� ������
            local_files_only=True      # <-- ���������, ��� ��� ������ ����!
        ).to("cuda")
        
        ml_models["sd_pipeline"] = pipe
        print("Lifespan: ������ ������� ��������� � ������ � ������!")

    except Exception as e:
        # !!! ��� ����� ������ ����� !!!
        # ���� ��� �������� ������ ���������� ����� ������, �� �� ������� � �����������
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ����������� ������ ��� �������� ������ !!!")
        print(f"!!! ��� ������: {type(e)}")
        print(f"!!! ����� ������: {e}")
        print("!!! ������ traceback ������:")
        traceback.print_exc() # ������������� ������ ��������� ������ � ����
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    yield
    
    # --- ���, ������� ����������� ��� ��������� ---
    print("Lifespan: ���������� ���������������. ������� �������.")
    ml_models.clear()


# ������� ���������� FastAPI � �������� ��� ���� ������� lifespan
app = FastAPI(lifespan=lifespan)

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    width: int = 512
    height: int = 512

@app.post("/generate")
def generate(body: RequestBody):
    # ���������, ����������� �� ������
    if "sd_pipeline" not in ml_models:
        return {"error": "������ �� ��������� ��-�� ������ ��� ������. ��������� ����."}

    # ����� ������ �� ������ ���������
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