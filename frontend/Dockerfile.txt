FROM python:3.11-slim

WORKDIR /app

# Устанавливаем библиотеки для фронта
RUN pip install --no-cache-dir streamlit requests plotly pandas

# Копируем код
COPY . .

# Открываем порт Streamlit
EXPOSE 8501

# Запускаем приложение
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]