### Запуск
```
docker-compose up -d
```

### После запуска
- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3000
 (логин admin / пароль admin)


### Проверка, что Prometheus видит метрики
По пути http://localhost:9090/targets должны отображаться два таргета (для triton и data_preprocessor)

### подключаем Grafana к Prometheus
В качестве Data Source выбираем Prometheus и указываем там URL `http://prometheus:9090`

