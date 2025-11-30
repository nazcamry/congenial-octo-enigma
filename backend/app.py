import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import shap
import os
import traceback

app = FastAPI(title="Alfa Income Predictor API")

# --- КОНСТАНТЫ ---
FALLBACK_INCOME = 92774.0  # Среднее по обучающей выборке
MODEL_PATH = "models/"

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
print("Загрузка моделей...")
try:
    cols_info = joblib.load(os.path.join(MODEL_PATH, "columns_info.pkl"))
    feat_cols = cols_info["all_columns"]
    cat_features = cols_info["cat_features"]

    cb_model = CatBoostRegressor()
    cb_model.load_model(os.path.join(MODEL_PATH, "catboost_final.cbm"))

    lgb_model = joblib.load(os.path.join(MODEL_PATH, "lightgbm_final.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_PATH, "xgboost_final.pkl"))
    meta_model = joblib.load(os.path.join(MODEL_PATH, "meta_model.pkl"))
    
    explainer = shap.TreeExplainer(cb_model)
    
    print("✅ Все модели успешно загружены!")
    
    # === ТЕСТ МАСШТАБА МОДЕЛЕЙ ===
    print("\n🔍 ПРОВЕРКА МАСШТАБА МОДЕЛЕЙ:")
    
    # Создаём тестовую строку (просто нули)
    test_df = pd.DataFrame([{col: 0 for col in feat_cols}])
    for col in cat_features:
        if col in test_df.columns:
            try:
                test_df[col] = test_df[col].astype('category')
            except:
                test_df[col] = test_df[col].astype(str).astype('category')
    
    # Проверяем что возвращают модели
    test_cb = cb_model.predict(test_df)[0]
    test_lgb = lgb_model.predict(test_df)[0]
    test_xgb = xgb_model.predict(xgb.DMatrix(test_df, enable_categorical=True))[0]
    
    print(f"   CatBoost raw: {test_cb:.2f}")
    print(f"   LightGBM raw: {test_lgb:.2f}")
    print(f"   XGBoost raw:  {test_xgb:.2f}")
    
    if test_cb > 1000:  # Если больше 1000, то это УЖЕ рубли, а не LOG
        print("   ⚠️  WARNING: Модели возвращают РУБЛИ, а не LOG!")
        print("   ⚠️  Отключаю np.expm1() для базовых моделей")
        MODELS_IN_LOG_SCALE = False
    else:
        print("   ✅ OK: Модели в LOG scale")
        MODELS_IN_LOG_SCALE = True
    print()
    
except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}")
    traceback.print_exc()
    MODELS_IN_LOG_SCALE = True  # По умолчанию


class ClientData(BaseModel):
    features: dict 


def get_recommendations(income):
    """Генерация рекомендаций продуктов на основе дохода"""
    recs = []
    
    if income < 50000:
        recs.append({
            "product": "Кредитная карта '100 дней без %'",
            "icon": "💳",
            "desc": "Лимит до 100 000 ₽",
            "priority": "high",
            "category": "Кредитные продукты"
        })
        recs.append({
            "product": "Накопительный Альфа-Счет",
            "icon": "💰",
            "desc": "До 16% годовых на остаток",
            "priority": "medium",
            "category": "Сбережения"
        })
        recs.append({
            "product": "Дебетовая карта с кэшбэком",
            "icon": "🎁",
            "desc": "До 10% кэшбэк в категориях",
            "priority": "medium",
            "category": "Повседневное"
        })
        
    elif 50000 <= income < 150000:
        recs.append({
            "product": "Кредит наличными",
            "icon": "💵",
            "desc": "Ставка от 4.5% годовых",
            "priority": "high",
            "category": "Кредитные продукты"
        })
        recs.append({
            "product": "Альфа-Вклад",
            "icon": "📈",
            "desc": "Максимальная доходность",
            "priority": "high",
            "category": "Сбережения"
        })
        recs.append({
            "product": "Автокредит",
            "icon": "🚗",
            "desc": "Одобрение за 30 минут",
            "priority": "medium",
            "category": "Крупные покупки"
        })
        recs.append({
            "product": "Страхование имущества",
            "icon": "🛡️",
            "desc": "Защита квартиры и авто",
            "priority": "low",
            "category": "Страхование"
        })
        
    else:  # income >= 150000
        recs.append({
            "product": "Премиум обслуживание Alfa Private",
            "icon": "💎",
            "desc": "Персональный менеджер 24/7",
            "priority": "high",
            "category": "Премиум"
        })
        recs.append({
            "product": "Инвестиционный портфель",
            "icon": "📊",
            "desc": "Акции, облигации, фонды",
            "priority": "high",
            "category": "Инвестиции"
        })
        recs.append({
            "product": "Ипотека",
            "icon": "🏠",
            "desc": "Одобрение за 1 минуту",
            "priority": "high",
            "category": "Крупные покупки"
        })
        recs.append({
            "product": "Металлический счет",
            "icon": "🥇",
            "desc": "Инвестиции в золото/серебро",
            "priority": "medium",
            "category": "Инвестиции"
        })
        recs.append({
            "product": "Страхование жизни",
            "icon": "❤️",
            "desc": "Защита семьи и накопления",
            "priority": "medium",
            "category": "Страхование"
        })
    
    return recs


@app.get("/")
def root():
    """Главная страница API"""
    return {
        "service": "Alfa Income Predictor API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "/health": "Проверка работоспособности",
            "/predict": "Прогноз дохода клиента (POST)",
            "/docs": "Интерактивная документация"
        }
    }


@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "ok",
        "service": "Alfa Income Predictor",
        "models_loaded": True,
        "models_in_log_scale": MODELS_IN_LOG_SCALE
    }


@app.post("/predict")
def predict_income(data: ClientData):
    """
    Прогноз дохода клиента на основе его характеристик
    
    Returns:
        - predicted_income: Прогнозируемый доход в рублях
        - confidence: Уровень уверенности (High/Medium/Low)
        - confidence_score: Числовой показатель уверенности (0-100)
        - base_models_pred: Прогнозы отдельных моделей
        - top_features: Топ-5 факторов, влияющих на прогноз
        - recommendations: Рекомендуемые продукты
    """
    try:
        # 1. Создаем DataFrame с полным набором признаков
        full_features = {col: 0 for col in feat_cols}
        full_features.update(data.features)
        
        input_df = pd.DataFrame([full_features])
        input_df = input_df[feat_cols]

        # Приводим категориальные признаки к правильному типу
        for col in cat_features:
            if col in input_df.columns:
                try:
                    input_df[col] = input_df[col].astype('category')
                except:
                    input_df[col] = input_df[col].astype(str).astype('category')

        # 2. Предсказания базовых моделей
        pred_cb_raw = cb_model.predict(input_df)[0]
        pred_lgb_raw = lgb_model.predict(input_df)[0]
        
        dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
        pred_xgb_raw = xgb_model.predict(dmatrix)[0]
        
        # Конвертируем в рубли (если модели в LOG scale)
        if MODELS_IN_LOG_SCALE:
            pred_cb_real = float(np.expm1(pred_cb_raw))
            pred_lgb_real = float(np.expm1(pred_lgb_raw))
            pred_xgb_real = float(np.expm1(pred_xgb_raw))
        else:
            # Модели уже вернули рубли
            pred_cb_real = float(pred_cb_raw)
            pred_lgb_real = float(pred_lgb_raw)
            pred_xgb_real = float(pred_xgb_raw)

        # 3. Стекинг через мета-модель
        X_meta = np.array([[pred_cb_real, pred_lgb_real, pred_xgb_real]])
        final_income = meta_model.predict(X_meta)[0]
        
        # 4. Валидация результата
        final_income = float(final_income)
        
        # Проверка на адекватность
        if np.isnan(final_income) or np.isinf(final_income) or final_income < 0:
            print(f"⚠️  Некорректный прогноз: {final_income}, использую fallback")
            final_income = FALLBACK_INCOME
        
        # Ограничиваем минимум (по требованиям)
        final_income = max(20000.0, final_income)

        # 5. SHAP объяснения (в рублях)
        shap_values_log = explainer.shap_values(input_df)[0]
        base_value_log = explainer.expected_value
        
        # Конвертируем SHAP в рублёвое влияние
        top_shap = []
        pred_log_full = base_value_log + np.sum(shap_values_log)
        
        for col, shap_log in zip(feat_cols, shap_values_log):
            if MODELS_IN_LOG_SCALE:
                # Предсказание БЕЗ этого признака
                pred_log_without = pred_log_full - shap_log
                
                # Конвертируем в рубли
                pred_rub_full = np.expm1(pred_log_full)
                pred_rub_without = np.expm1(pred_log_without)
                
                impact_rub = pred_rub_full - pred_rub_without
            else:
                # Если модели не в LOG, используем сырой SHAP
                impact_rub = shap_log
            
            top_shap.append((col, float(impact_rub)))
        
        # Топ-5 по модулю влияния
        top_shap = sorted(top_shap, key=lambda x: abs(x[1]), reverse=True)[:5]

        # 6. Уверенность модели (улучшенная метрика)
        preds_real = [pred_cb_real, pred_lgb_real, pred_xgb_real]
        std_dev = np.std(preds_real)
        cv = std_dev / final_income if final_income > 0 else 1.0  # Коэффициент вариации
        
        if cv < 0.05:
            confidence = "High"
        elif cv < 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        confidence_score = max(0, min(100, int((1 - cv) * 100)))

        # 7. Рекомендации продуктов
        recommendations = get_recommendations(final_income)

        return {
            "predicted_income": round(final_income, 2),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "base_models": {
                "catboost": round(pred_cb_real, 2),
                "lightgbm": round(pred_lgb_real, 2),
                "xgboost": round(pred_xgb_real, 2),
                "ensemble_std": round(std_dev, 2)
            },
            "top_features": [
                {
                    "feature": feat,
                    "impact": round(impact, 2),
                    "impact_percent": round(abs(impact) / final_income * 100, 1) if final_income > 0 else 0
                }
                for feat, impact in top_shap
            ],
            "recommendations": recommendations
        }

    except Exception as e:
        print("=" * 60)
        print("❌ ОШИБКА ПРИ ПРЕДСКАЗАНИИ:")
        print(f"Input features: {data.features}")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при прогнозировании: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Запуск Alfa Income Predictor API")
    print("=" * 60)
    print("📍 API доступен по адресу: http://localhost:8000")
    print("📚 Документация: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)