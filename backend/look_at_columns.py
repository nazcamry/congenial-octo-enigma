import joblib
import os

# Путь к файлу
path = "models/columns_info.pkl"

if os.path.exists(path):
    data = joblib.load(path)
    cols = data["all_columns"]
    
    print("\n--- СПИСОК ПРИЗНАКОВ (Первые 30) ---")
    print(cols[:30])
    print("\n------------------------------------")
    print(f"Всего признаков: {len(cols)}")
    
    # Попробуем угадать ключевые
    print("\nПопробуй найти эти названия в списке выше:")
    print("- Возраст (age?)")
    print("- Стаж (experience / tenure?)")
    print("- Скор (score?)")
    print("- Регион (region / geo?)")
else:
    print("Не нашел файл models/columns_info.pkl")