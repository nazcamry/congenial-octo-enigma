import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==================================================================================
# НАСТРОЙКА СТРАНИЦЫ
# ==================================================================================

st.set_page_config(
    page_title="Alfa Income AI", 
    page_icon="🅰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #F2F3F5 0%, #E8EAED 100%);
    }
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #EF3124;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #EF3124 0%, #C91F1A 100%);
        color: white;
        border-radius: 10px;
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 8px rgba(239,49,36,0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #C91F1A 0%, #EF3124 100%);
        box-shadow: 0 6px 12px rgba(239,49,36,0.4);
        transform: translateY(-2px);
    }
    .info-box {
        background-color: #FFF3F3;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #EF3124;
        margin: 10px 0;
    }
    h1 {
        color: #2C3E50;
        font-weight: 700;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    </style>
    """, unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ==================================================================================
# САЙДБАР - ВЫБОР КЛИЕНТА
# ==================================================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Alfa_Bank.svg/1200px-Alfa_Bank.svg.png",
    width=200
)
st.sidebar.title("⚙️ Параметры клиента")

# Проверка API
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("🟢 API подключен")
    else:
        st.sidebar.error("🔴 API недоступен")
except:
    st.sidebar.error("🔴 API недоступен")
    st.sidebar.info("Запустите: `python app.py`")

st.sidebar.markdown("---")

# Выбор профиля клиента
client_type = st.sidebar.selectbox(
    "📋 Профиль клиента:",
    ["Молодой специалист", "Опытный менеджер", "Топ-менеджер", "Пенсионер", "Кастомный"],
    help="Выберите готовый профиль или создайте свой"
)

# ==================================================================================
# ПРЕСЕТЫ ДАННЫХ (это "база клиентов")
# ==================================================================================

CLIENT_PRESETS = {
    "Молодой специалист": {
        "age": 25,
        "region": 77,  # Москва
        "salary_avg": 55000,
        "turnover": 35000,
        "credit_limit": 80000,
        "balance": 15000,
        "work_experience": 3,
        "education": "Высшее",
        "family_status": "Холост/не замужем"
    },
    "Опытный менеджер": {
        "age": 38,
        "region": 78,  # Санкт-Петербург
        "salary_avg": 145000,
        "turnover": 95000,
        "credit_limit": 400000,
        "balance": 250000,
        "work_experience": 15,
        "education": "Высшее + MBA",
        "family_status": "Женат/замужем"
    },
    "Топ-менеджер": {
        "age": 48,
        "region": 77,
        "salary_avg": 950000,
        "turnover": 650000,
        "credit_limit": 2000000,
        "balance": 4500000,
        "work_experience": 25,
        "education": "Высшее + MBA",
        "family_status": "Женат/замужем"
    },
    "Пенсионер": {
        "age": 67,
        "region": 54,  # Новосибирск
        "salary_avg": 28000,
        "turnover": 18000,
        "credit_limit": 0,
        "balance": 350000,
        "work_experience": 40,
        "education": "Высшее",
        "family_status": "Вдовец/вдова"
    },
    "Кастомный": {
        "age": 30,
        "region": 77,
        "salary_avg": 80000,
        "turnover": 50000,
        "credit_limit": 150000,
        "balance": 50000,
        "work_experience": 7,
        "education": "Высшее",
        "family_status": "Холост/не замужем"
    }
}

selected_profile = CLIENT_PRESETS[client_type]

# ==================================================================================
# ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ
# ==================================================================================

st.sidebar.markdown("### 🎯 Основные параметры")

if client_type == "Кастомный":
    age = st.sidebar.slider("Возраст", 18, 80, selected_profile["age"])
    region = st.sidebar.number_input("Регион (код)", 1, 99, selected_profile["region"])
    salary_avg = st.sidebar.number_input("Средний доход (₽)", 0, 2000000, selected_profile["salary_avg"], step=5000)
else:
    age = st.sidebar.slider("Возраст", 18, 80, selected_profile["age"])
    region = selected_profile["region"]
    salary_avg = selected_profile["salary_avg"]

# Коэффициент активности
activity_mult = st.sidebar.slider(
    "📊 Финансовая активность",
    0.5, 2.0, 1.0, 0.1,
    help="Влияет на обороты по картам и транзакции"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Информация о профиле")
st.sidebar.info(f"""
**Образование:** {selected_profile['education']}  
**Стаж:** {selected_profile['work_experience']} лет  
**Семья:** {selected_profile['family_status']}
""")

predict_btn = st.sidebar.button("🚀 РАССЧИТАТЬ ДОХОД", use_container_width=True)

# ==================================================================================
# ГЛАВНЫЙ ЭКРАН
# ==================================================================================

# Заголовок
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Alfa_Bank.svg/1200px-Alfa_Bank.svg.png", width=100)
with col_title:
    st.title("💸 AI-Прогноз Дохода Клиента")
    st.markdown("*Интеллектуальная система оценки платёжеспособности на базе ML*")

st.markdown("---")

# Инфо о клиенте
with st.expander("ℹ️ О системе", expanded=False):
    st.markdown("""
    **Система использует:**
    - 🤖 Ансамбль из 3 моделей: CatBoost + LightGBM + XGBoost
    - 📊 150+ признаков: транзакции, БКИ, цифровой профиль
    - 🎯 WMAE метрика для оценки точности
    - 🔍 SHAP для объяснимости прогнозов
    
    **Бизнес-применение:**
    - Скоринг при выдаче кредитов
    - Персонализация продуктовых предложений
    - Сегментация клиентской базы
    """)

# ==================================================================================
# ОБРАБОТКА ЗАПРОСА
# ==================================================================================

if predict_btn:
    with st.spinner('🔄 Анализ транзакционного профиля клиента...'):
        
        # Формируем признаки на основе профиля
        base_salary = salary_avg * activity_mult
        base_turnover = selected_profile["turnover"] * activity_mult
        
        # Создаём payload с КЛЮЧЕВЫМИ признаками
        # Остальные признаки API заполнит нулями автоматически
        payload = {
            "features": {
                # Демографические
                "age": age,
                "adminarea": str(region),
                "gender": 1,
                
                # Доходы (самые важные!)
                "salary_6to12m_avg": base_salary,
                "dp_ils_avg_salary_1y": base_salary * 0.95,  # Немного варьируем
                "incomeValue": base_salary * 1.05,
                "avg_salary_3m": base_salary * 0.9,
                
                # Транзакции и обороты
                "turn_cur_cr_avg_v2": base_turnover,
                "avg_cur_cr_turn": base_turnover * 1.1,
                "turn_cur_cr_sum_v2": base_turnover * 12,
                
                # Кредитная история
                "hdb_bki_total_max_limit": selected_profile["credit_limit"],
                "hdb_bki_active_cc_max_limit": selected_profile["credit_limit"] * 0.6,
                "bki_max_limit": selected_profile["credit_limit"] * 0.8,
                
                # Балансы
                "curr_rur_amt_cm_avg": selected_profile["balance"],
                "avg_balance": selected_profile["balance"] * 0.9,
                
                # Опыт работы
                "work_experience_years": selected_profile["work_experience"],
            }
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # ============================================================
                # ОСНОВНЫЕ МЕТРИКИ
                # ============================================================
                st.success("✅ Прогноз успешно рассчитан!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💰 Прогнозируемый доход",
                        f"{data['predicted_income']:,.0f} ₽",
                        help="Оценка ежемесячного дохода клиента"
                    )
                
                with col2:
                    confidence_emoji = {
                        "High": "🟢",
                        "Medium": "🟡",
                        "Low": "🔴"
                    }
                    st.metric(
                        "🎯 Уверенность модели",
                        f"{confidence_emoji.get(data['confidence'], '⚪')} {data['confidence']}",
                        f"{data['confidence_score']}%",
                        help="Насколько модель уверена в прогнозе"
                    )
                
                with col3:
                    # Разброс прогнозов базовых моделей
                    base_models = data['base_models']
                    preds = [base_models['catboost'], base_models['lightgbm'], base_models['xgboost']]
                    spread = max(preds) - min(preds)
                    
                    st.metric(
                        "📊 Разброс моделей",
                        f"±{spread/2:,.0f} ₽",
                        help="Разница между прогнозами моделей"
                    )
                
                with col4:
                    # Примерная категория дохода
                    income = data['predicted_income']
                    if income < 50000:
                        category = "💼 Начальный"
                    elif income < 150000:
                        category = "📈 Средний+"
                    else:
                        category = "💎 Премиум"
                    
                    st.metric(
                        "👤 Сегмент клиента",
                        category,
                        help="Категория по уровню дохода"
                    )
                
                st.markdown("---")
                
                # ============================================================
                # ДЕТАЛИЗАЦИЯ МОДЕЛЕЙ
                # ============================================================
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("🔍 Факторы влияния на прогноз (SHAP)")
                    
                    # Красивые названия признаков
                    FEATURE_NAMES = {
                        "salary_6to12m_avg": "📊 Средняя зарплата (6-12 мес)",
                        "age": "🎂 Возраст клиента",
                        "hdb_bki_total_max_limit": "💳 Макс. кредитный лимит",
                        "turn_cur_cr_avg_v2": "🔄 Средний оборот по картам",
                        "incomeValue": "💰 Оценка дохода (БКИ)",
                        "curr_rur_amt_cm_avg": "🏦 Средний остаток на счете",
                        "dp_ils_avg_salary_1y": "📈 ЗП за последний год",
                        "work_experience_years": "👔 Стаж работы",
                        "avg_cur_cr_turn": "💸 Обороты (среднее)",
                        "bki_max_limit": "📋 Лимит по БКИ"
                    }
                    
                    top_features = data['top_features']
                    
                    # Подготовка данных для графика
                    features = []
                    impacts = []
                    colors = []
                    
                    for feat_data in top_features:
                        feat_name = feat_data['feature']
                        impact = feat_data['impact']
                        
                        # Красивое название
                        display_name = FEATURE_NAMES.get(feat_name, feat_name)
                        features.append(display_name)
                        impacts.append(impact)
                        colors.append('#00B92D' if impact > 0 else '#EF3124')
                    
                    # График Waterfall
                    fig = go.Figure(go.Bar(
                        x=impacts,
                        y=features,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{val:+,.0f} ₽" for val in impacts],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Влияние: %{x:,.0f} ₽<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Влияние на прогноз (₽)",
                        yaxis_title="",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='white',
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Текстовое объяснение
                    with st.expander("💡 Что это значит?"):
                        st.markdown(f"""
                        **Как читать график:**
                        - 🟢 **Зелёные** столбцы **увеличивают** прогноз дохода
                        - 🔴 **Красные** столбцы **уменьшают** прогноз дохода
                        - Длина столбца показывает **силу влияния** признака
                        
                        **В данном случае:**
                        - Самый важный фактор: **{FEATURE_NAMES.get(top_features[0]['feature'], top_features[0]['feature'])}**
                        - Его вклад: **{top_features[0]['impact']:+,.0f} ₽** ({top_features[0]['impact_percent']}%)
                        """)
                
                with col_right:
                    st.subheader("🤖 Прогнозы моделей")
                    
                    # Сравнение моделей
                    models_df = pd.DataFrame({
                        'Модель': ['CatBoost', 'LightGBM', 'XGBoost', 'Ансамбль'],
                        'Прогноз (₽)': [
                            base_models['catboost'],
                            base_models['lightgbm'],
                            base_models['xgboost'],
                            data['predicted_income']
                        ]
                    })
                    
                    fig_models = px.bar(
                        models_df,
                        x='Модель',
                        y='Прогноз (₽)',
                        color='Модель',
                        color_discrete_sequence=['#EF3124', '#FFA500', '#00B92D', '#1E90FF'],
                        text='Прогноз (₽)'
                    )
                    
                    fig_models.update_traces(
                        texttemplate='%{text:,.0f} ₽',
                        textposition='outside'
                    )
                    
                    fig_models.update_layout(
                        height=350,
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_models, use_container_width=True)
                    
                    st.info(f"""
                    **Стандартное отклонение:**  
                    {base_models['ensemble_std']:,.0f} ₽
                    
                    *Чем меньше разброс, тем выше уверенность*
                    """)
                
                st.markdown("---")
                
                # ============================================================
                # РЕКОМЕНДАЦИИ ПРОДУКТОВ
                # ============================================================
                st.subheader("🎁 Персональные предложения")
                
                recommendations = data['recommendations']
                
                # Группируем по категориям
                categories = {}
                for rec in recommendations:
                    cat = rec.get('category', 'Другое')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(rec)
                
                # Показываем по категориям
                for category, recs in categories.items():
                    st.markdown(f"**{category}**")
                    cols = st.columns(min(len(recs), 3))
                    
                    for i, rec in enumerate(recs):
                        with cols[i % 3]:
                            priority_color = {
                                'high': '🔥',
                                'medium': '⭐',
                                'low': '💡'
                            }
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{rec['icon']} {rec['product']}</h3>
                                <p style="color: #666; margin: 10px 0;">{rec['desc']}</p>
                                <p style="margin: 0;"><strong>Приоритет:</strong> {priority_color.get(rec.get('priority', 'medium'), '⭐')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Ошибка сервера ({response.status_code})")
                with st.expander("🔍 Детали ошибки"):
                    st.code(response.text)
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Превышено время ожидания. Попробуйте ещё раз.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Не удалось подключиться к API. Убедитесь, что сервер запущен.")
            st.info("Запустите API командой: `python app.py`")
        except Exception as e:
            st.error(f"❌ Непредвиденная ошибка: {str(e)}")
            with st.expander("🐛 Детали для отладки"):
                st.code(str(e))

else:
    # Плейсхолдер, пока не нажата кнопка
    st.info("👈 Выберите профиль клиента и нажмите **'РАССЧИТАТЬ ДОХОД'**")
    
    # Демо-статистика
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 О системе")
        st.markdown("""
        - **Точность:** WMAE 61,645₽
        - **Моделей:** 3 (ансамбль)
        - **Признаков:** 150+
        """)
    
    with col2:
        st.markdown("### 🎯 Применение")
        st.markdown("""
        - Кредитный скоринг
        - Персонализация
        - Fraud detection
        """)
    
    with col3:
        st.markdown("### 💡 Преимущества")
        st.markdown("""
        - Быстрый прогноз (<1 сек)
        - Объяснимость (SHAP)
        - Авто-рекомендации
        """)

# ==================================================================================
# FOOTER
# ==================================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Alfa Income AI Predictor</strong> | Hack&Change 2025</p>
    <p style='font-size: 12px;'>Powered by CatBoost + LightGBM + XGBoost | FastAPI + Streamlit</p>
</div>
""", unsafe_allow_html=True)