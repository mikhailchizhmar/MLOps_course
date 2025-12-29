```
docker run -d --name model model_image  
docker run -d --name preprocess --link model:model preprocess_image  
docker run -d --name postprocess --link preprocess:preprocess postprocess_image  
```
```
docker compose up --build  
```