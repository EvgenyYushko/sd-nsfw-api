# --- ���� 1: "������� ������" ---
FROM python:3.10-slim as builder

# ��������� ������������ ��� ����������
# git � git-lfs ����� ��� ���������� ��������� ������� � HF
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors

# ��������� �������� ����� ��� ���� huggingface
# �����: ��� �� �����, ������� �� ����� ���������
ENV HF_HOME=/hf_cache

# �������� � ��������� ��� ������ ��� ����������
COPY download_model.py .
RUN python download_model.py

# --- ���� 2: "��������� ����� ����������" ---
FROM python:3.10-slim

WORKDIR /app

# ������������� ������ �� �����������, ��� ����� ��� ������ API
# PyTorch � diffusers ��� ����� �����������, �� ����� ������� �� ����
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow

# �������� ��� ����� � ����� ������ �� �������� � ��������� �����
# ��� ����� ������ ���!
COPY --from=builder /hf_cache /root/.cache/huggingface

# �������� �������� ��� ����������
COPY app.py .

# ������� ��� ������� ���-�������. ����������� ���������� PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]