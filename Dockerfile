FROM python:3.13-slim-bookworm AS tracker

RUN pip install --no-cache-dir mlflow==3.18.0

FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS train

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY prepare_data.py train.py data.yaml ./
CMD ["python", "train.py"]

FROM runtime AS api
