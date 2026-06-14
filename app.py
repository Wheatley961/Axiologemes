import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==============================================================================
st.set_page_config(
    page_title="Анализ семантического пространства аксиологем",
    page_icon="🧠",
    layout="wide"
)

# ==============================================================================
# ГЕНЕРАЦИЯ ДЕМО-ДАННЫХ (если файл не загружен)
# ==============================================================================
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    axiologemes = ["Свобода", "Справедливость", "Честь", "Добро", "Истина", "Власть"]
    years = [2010, 2015, 2020, 2025]
    
    data = []
    for ax in axiologemes:
        for year in years:
            # Генерируем "кластер" точек вокруг условного центра аксиологемы
            base_x = np.random.uniform(-5, 5)
            base_y = np.random.uniform(-5, 5)
            base_z = np.random.uniform(-5, 5)
            
            for _ in range(15): # по 15 контекстных употреблений на год
                data.append({
                    "word": f"{ax}_context_{np.random.randint(1, 100)}",
                    "axiologeme": ax,
                    "year": year,
                    "x": base_x + np.random.normal(0, 1.5),
                    "y": base_y + np.random.normal(0, 1.5),
                    "z": base_z + np.random.normal(0, 1.5)
                })
    return pd.DataFrame(data)

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ
# ==============================================================================
st.sidebar.title("⚙️ Настройки")
st.sidebar.markdown("Загрузите CSV-файл со столбцами: `word`, `axiologeme`, `year`, `x`, `y`, `z`")
uploaded_file = st.sidebar.file_uploader("Загрузить данные", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Файл успешно загружен!")
else:
    df = generate_demo_data()
    st.sidebar.info("Используются демонстрационные данные.")

# ==============================================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ==============================================================================
st.title("🧠 Семантический анализ аксиологем")
st.markdown("Интерактивная визуализация семантического пространства и оценка смысловой близости концептов.")

tabs = st.tabs(["📊 3D Визуализация", "📐 Калькулятор близости", "📚 Документация и формулы"])

# ------------------------------------------------------------------------------
# ВКЛАДКА 1: 3D ВИЗУАЛИЗАЦИЯ (Без анимации центроидов по годам)
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Пространственное распределение аксиологем")
    
    # Фильтр по годам (статический выбор, без анимации дрейфа)
    available_years = sorted(df["year"].unique())
    selected_years = st.multiselect("Выберите годы для отображения:", available_years, default=available_years)
    
    filtered_df = df[df["year"].isin(selected_years)]
    
    if not filtered_df.empty:
        fig = px.scatter_3d(
            filtered_df,
            x="x", y="y", z="z",
            color="axiologeme",
            hover_data=["word", "year"],
            title="Семантическое пространство аксиологем (3D)",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Настройка внешнего вида графика
        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            scene=dict(
                xaxis_title="Компонента 1 (X)",
                yaxis_title="Компонента 2 (Y)",
                zaxis_title="Компонента 3 (Z)"
            ),
            legend_title="Аксиологема"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("💡 **Подсказка:** Наведите курсор на точку, чтобы увидеть конкретное слово и год. Используйте мышь для вращения 3D-сцены.")
    else:
        st.warning("Нет данных для выбранных критериев.")

# ------------------------------------------------------------------------------
# ВКЛАДКА 2: КАЛЬКУЛЯТОР СЕМАНТИЧЕСКОЙ БЛИЗОСТИ
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Оценка косинусной близости аксиологем")
    st.markdown("Выберите две аксиологемы для сравнения их усреднённых векторных представлений в текущем наборе данных.")
    
    unique_ax = sorted(df["axiologeme"].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        ax1 = st.selectbox("Первая аксиологема", unique_ax, index=0)
    with col2:
        ax2 = st.selectbox("Вторая аксиологема", unique_ax, index=1 if len(unique_ax) > 1 else 0)
    
    if st.button("Рассчитать близость"):
        # Расчёт центроидов (статический, для выбранных аксиологем)
        vec1 = df[df["axiologeme"] == ax1][["x", "y", "z"]].mean(axis=0).values.reshape(1, -1)
        vec2 = df[df["axiologeme"] == ax2][["x", "y", "z"]].mean(axis=0).values.reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        
        st.metric(
            label=f"Косинусная близость: '{ax1}' и '{ax2}'",
            value=f"{similarity:.4f}",
            delta="Высокая семантическая связь" if similarity > 0.7 else "Умеренная/Низкая связь"
        )
        
        # Визуализация векторов
        st.markdown("#### Векторные координаты (усреднённые)")
        res_df = pd.DataFrame({
            "Аксиологема": [ax1, ax2],
            "X": [float(vec1[0][0]), float(vec2[0][0])],
            "Y": [float(vec1[0][1]), float(vec2[0][1])],
            "Z": [float(vec1[0][2]), float(vec2[0][2])]
        })
        st.dataframe(res_df, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# ВКЛАДКА 3: ДОКУМЕНТАЦИЯ И ФОРМУЛЫ
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Математические основы анализа")
    
    st.markdown("""
    ### 1. Векторное представление (Word Embeddings)
    Каждая лексическая единица представляется как вектор в многомерном пространстве $\mathbb{R}^d$:
    """)
    st.latex(r"\vec{v}_w = [v_1, v_2, \dots, v_d]")
    st.markdown("Где $d$ – размерность эмбеддинга (например, 300 для моделей Word2Vec или FastText).")

    st.markdown("### 2. Снижение размерности (PCA)")
    st.markdown("Для визуализации в 3D-пространстве исходные векторы проецируются с помощью метода главных компонент:")
    st.latex(r"C = \frac{1}{N} \sum_{i=1}^{N} (\vec{x}_i - \vec{\mu})(\vec{x}_i - \vec{\mu})^T")
    st.markdown("Где $C$ – ковариационная матрица, а $\vec{\mu}$ – вектор средних значений. Проекция осуществляется на 3 собственных вектора, соответствующих наибольшим собственным числам.")

    st.markdown("### 3. Косинусная мера сходства")
    st.markdown("Оценивает угол между векторами двух аксиологем $\vec{A}$ и $\vec{B}$:")
    st.latex(r"\text{CosineSimilarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}")
    st.markdown("Значение $1.0$ означает полную сонаправленность (семантическую идентичность в данном контексте), $0.0$ – ортогональность (независимость), $-1.0$ – противоположность.")
    
    st.info("📌 **Примечание:** Функционал анимированного расчёта центроидов по годам был удалён в соответствии с требованиями, однако статический анализ пространственного распределения и попарное сравнение сохранены и расширены.")

# ==============================================================================
# ФУТЕР
# ==============================================================================
st.markdown("---")
st.caption("Семантический анализатор аксиологем | Streamlit Application")
