FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.10-slim
COPY --from=base /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY model.onnx inference.py /app/
WORKDIR /app
CMD ["python", "inference.py"]
