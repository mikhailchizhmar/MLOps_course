# postprocess/app.py
from fastapi import FastAPI, Request
import requests, numpy as np

app = FastAPI()

@app.post("/finalize")
async def finalize(req: Request):
    data = await req.json()
    r = requests.post("http://preprocess:5000/process", files={"file": open("image.jpg", "rb")})
    result = r.json()["result"]
    label = int(np.argmax(result))
    return {"class_id": label}
