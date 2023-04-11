## Уcтановка 

### Создание и активация виртуального окружения
`python3 -m venv venv`

`source venv/bin/activate` (для Windows иначе)

### Установка зависимостей
`pip install -r requirements.txt`

### Запуск приложения
`python app.py`

### Тестирование
`curl -i -X POST -H "Content-Type: multipart/form-data" -F "file=@img.jpg" http://localhost:8080/predict`