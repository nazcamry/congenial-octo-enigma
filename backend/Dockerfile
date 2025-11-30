FROM python:3.11-slim

# ШАГ 1: Устанавливаем системную библиотеку для LightGBM (ОБЯЗАТЕЛЬНО)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ШАГ 2: Копируем файлы
COPY . .

# ШАГ 3: Устанавливаем Python-библиотеки
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn catboost lightgbm xgboost shap joblib

EXPOSE 8000

# ШАГ 4: Запускаем
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]