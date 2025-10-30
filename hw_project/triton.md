## Triton Inference Server

### Запуск
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `
-v C:\Users\chizh\Desktop\Магистратура\MLOps_course\hw_project\model_repository:/models `
nvcr.io/nvidia/tritonserver:24.08-py3 `
tritonserver --model-repository=/models --backend-config=onnxruntime,device=CPU --backend-config=pytorch,device=CPU
```

### Проверка

```
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/model_optim_pt
```

Ответ сервера на второй запрос: 
```
{"name":"model_optim_pt","versions":["1"],"platform":"pytorch_libtorch","inputs":[{"name":"INPUT__0","datatype":"FP32","shape":[-1,3,64,64]}],"outputs":[{"name":"OUTPUT__0","datatype":"FP32","shape":[-1,5]}]}
```