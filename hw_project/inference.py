import requests
import numpy as np
import tritonclient.http as httpclient


def run_inference(image_path: str):
    # 1️⃣ Отправляем изображение на FastAPI микросервис
    with open(image_path, "rb") as f:
        files = {"file": f}
        resp = requests.post("http://localhost:8050/preprocess", files=files)
    resp.raise_for_status()
    data = np.array(resp.json()["data"], dtype=np.float32)

    # 2️⃣ Подключаемся к Triton
    client = httpclient.InferenceServerClient(url="localhost:8000")

    inputs = httpclient.InferInput("INPUT__0", data.shape, "FP32")
    inputs.set_data_from_numpy(data)

    outputs = httpclient.InferRequestedOutput("OUTPUT__0")

    # 3️⃣ Запрос на инференс
    result = client.infer(model_name="model_optim_pt", inputs=[inputs], outputs=[outputs])

    # 4️⃣ Получаем результат
    output_data = result.as_numpy("OUTPUT__0")
    pred_class = int(np.argmax(output_data))
    prob = float(np.max(output_data))
    selected_classes = ["chicken_wings", "pizza", "french_fries", "hamburger", "sushi"]

    print(f"✅ Предсказанный класс: {selected_classes[pred_class]}")
    return pred_class


if __name__ == 'main':
    run_inference('img/pizza.jpg')
    run_inference('img/french_fries.jpg')
