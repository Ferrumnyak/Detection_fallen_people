from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, ImageOps
from PIL.Image import open as open_image
from ultralytics import YOLO, settings

WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
FALL_CLASS_ID = 1
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
prediction_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Сначала обучите модель. Нужен файл {WEIGHTS}")

    settings.update({"sync": False})
    model = YOLO(str(WEIGHTS))
    if model.task != "detect" or model.names.get(FALL_CLASS_ID) != "fallen_person":
        raise ValueError("Нужны веса YOLO11s, обученные на FPDS")

    app.state.model = model  # Сохраняем модель в общей памяти приложения
    yield  # Позволяет функции приостановиться на этой строке и продолжиться позже


app = FastAPI(title="Детекция упавших людей", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Изображение JPEG или PNG")],
    confidence: Annotated[float, Query(gt=0, le=1)] = 0.25,
):
    content = file.file.read(MAX_FILE_SIZE + 1)  # Содержимое файла в байтах
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "Файл должен весить не больше 10 МБ")

    try:
        # BytesIO делает из полученных байтов объект, который можно читать как файл
        # with закроет source при выходе
        with open_image(BytesIO(content), formats=("JPEG", "PNG")) as source:
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "В изображении больше 16 млн пикселей")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "Изображение слишком большое") from error
    except OSError as error:
        raise HTTPException(
            400, "Загрузите корректное изображение JPEG или PNG"
        ) from error

    with prediction_lock:
        result = request.app.state.model.predict(
            image, classes=[FALL_CLASS_ID], conf=confidence, device="cpu", verbose=False
        )[0]

    detections = []
    for box in result.boxes:
        detections.append(
            {
                "class": "fallen_person",
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(value, 2) for value in box.xyxy[0].tolist()],
            }
        )
    return {"has_fall": bool(detections), "detections": detections}
