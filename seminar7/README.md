```
name: "my_tf_model"
platform: "tensorflow_savedmodel"
max_batch_size: 64
input [
  {
    name: "input_tensor"
    data_type: TYPE_FP32
    dims: [ -1, 224, 224, 3 ]
  }
]
output [
  {
    name: "output_tensor"
    data_type: TYPE_FP32
    dims: [ -1, 1000 ]
  }
]
```

```
name: "my_onnx_model"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

[https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)  


curl -s -X POST -F "file=@cat.jpg" http://localhost:8001/process  

[https://github.com/triton-inference-server/server/releases](https://github.com/triton-inference-server/server/releases)  

[https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html)


[Пример модели, которая успешно запускается в trition](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-qdq.onnx)