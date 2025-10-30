from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import io

# Инициализация приложения
app = FastAPI(title="Data Preprocessing Service",
              description="Микросервис для предобработки изображений перед подачей в CNN",
              version="1.0")

# Трансформации как при обучении модели
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """
    Принимает изображение, применяет трансформации и возвращает тензор (в JSON виде).
    """
    try:
        # читаем байты и открываем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # применяем те же трансформации, что и при обучении
        tensor = transform(image).unsqueeze(0)  # добавим batch dimension [1,3,64,64]

        # конвертируем в список для JSON
        data = tensor.tolist()
        return JSONResponse(content={"status": "ok", "tensor_shape": list(tensor.shape), "data": data[:1]})
    
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)})


@app.get("/")
def root():
    return {"message": "Сервис предобработки данных работает. Отправь POST /preprocess с изображением."}
