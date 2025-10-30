## Инструкция по запуску на Windows

### Локальный запуск
```
uvicorn app:app --reload
```

### Docker
1. Открываем Docker Desktop
2. Убеждаемся, что в левом углу написано "Engine running"
3. `docker build -t data-preprocessor .`
4. `docker run -d -p 8050:8000 data-preprocessor`
