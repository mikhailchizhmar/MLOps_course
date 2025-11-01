from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from torchvision import transforms
from PIL import Image
import io
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# Инициализация приложения
app = FastAPI(title="Data Preprocessing Service",
              description="Микросервис для предобработки изображений перед подачей в CNN",
              version="1.0")

# Трансформации
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# METRICS
REQUEST_COUNTER = Counter("dp_requests_total", "Total requests to data_preprocessor")
PROCESS_TIME = Histogram("dp_request_processing_seconds", "Processing time for requests in seconds")


@app.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """
    Принимает изображение, применяет трансформации и возвращает тензор (в JSON виде).
    """
    start = time.time()
    REQUEST_COUNTER.inc()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # добавим batch dimension [1,3,64,64]
        data = tensor.tolist()
        duration = time.time() - start
        PROCESS_TIME.observe(duration)
        return JSONResponse(content={"status": "ok", "tensor_shape": list(tensor.shape), "data": data[:1]})
    except Exception as e:
        duration = time.time() - start
        PROCESS_TIME.observe(duration)
        return JSONResponse(content={"status": "error", "detail": str(e)})


# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    # generate_latest uses the default registry
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {"message": "Data Preprocessor running. POST /preprocess with image."}