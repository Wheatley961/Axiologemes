import streamlit as st
import pandas as pd
import numpy as np
import re
import itertools
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull, Delaunay
import plotly.express as px
import plotly.graph_objects as go
import stanza
import os

# ==============================================================================
# НАСТРОЙКИ СТРАНИЦЫ
# ==============================================================================
st.set_page_config(page_title="Анализ Аксиологем", page_icon="📊", layout="wide")

# ==============================================================================
# КЭШИРОВАНИЕ РЕСУРСОВ И ДАННЫХ
# ==============================================================================
@st.cache_resource
def load_stanza_model():
    """Загрузка модели Stanza для лемматизации (кэшируется)"""
    stanza.download("ru", verbose=False)
    return stanza.Pipeline(lang="ru", processors="tokenize,pos,lemma", use_gpu=False)

@st.cache_resource
def load_embedding_model():
    """Загрузка модели векторизации текста (кэшируется)"""
    return SentenceTransformer('cointegrated/rubert-tiny2') # Лёгкая и быстрая модель для примера

@st.cache_data
def process_data(uploaded_files):
    """Загрузка и предварительная обработка данных из Excel"""
    dfs = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    if not dfs:
        return None
        
    df = pd.concat(dfs, ignore_index=True).fillna("")
    
    # Формирование полного контекста
    df["full_context"] = (
        df["Left context"].astype(str) + " " +
        df["Center"].astype(str) +
        df["Punct"].astype(str) + " " +
        df["Right context"].astype(str)
    )
    # Очистка пробелов
    df["full_context"] = df["full_context"].str.replace(r"\s+", " ", regex=True)
    
    return df

@st.cache_data
def get_embeddings_and_pca(texts, n_components=2):
    """Векторизация текстов и снижение размерности до 2D"""
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    return embeddings, coords

def lemmatize_text(text, nlp):
    """Лемматизация текста с помощью Stanza"""
    doc = nlp(text)
    return " ".join([word.lemma.lower() for sent in doc.sentences for word in sent.words if word.pos in ['NOUN', 'ADJ', 'VERB']])

@st.cache_data
def get_top_words(df, nlp, top_n=50):
    """Извлечение топ-N слов для каждой аксиологемы"""
    top_words = {}
    for ax in df["axiologeme"].unique():
        subset = df[df["axiologeme"] == ax]
        lemmatized = subset["full_context"].apply(lambda x: lemmatize_text(x, nlp))
        all_words = " ".join(lemmatized).split()
        # Простой подсчёт частоты (можно усложнить до TF-IDF)
        word_counts = pd.Series(all_words).value_counts()
        # Исключаем саму аксиологему и стоп-слова (упрощённо)
        stop_words = {ax.lower(), "который", "этот", "тот", "быть", "мочь"}
        filtered_words = [w for w, c in word_counts.items() if w not in stop_words and len(w) > 3]
        top_words[ax] = filtered_words[:top_n]
    return top_words

def points_in_hull_2d(points, hull_points):
    """Проверка, какие точки лежат внутри 2D выпуклой оболочки"""
    if len(hull_points) < 3:
        return np.array([False] * len(points))
    hull = Delaunay(hull_points)
    return hull.find_simplex(points) >= 0

# ==============================================================================
# ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ==============================================================================
st.title("📊 Семантический анализ аксиологем")
st.markdown("""
Это приложение анализирует семантические поля, динамику упоминаний, пересечения и внутреннюю полисемию аксиологем (например, "болезнь" и "здоровье") на основе текстовых корпусов.
Все 3D-визуализации заменены на информативные 2D-графики для удобства восприятия.
""")

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("⚙️ Загрузка данных")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите Excel файлы (ruscorpora_content_*.xlsx)", 
    type=["xlsx"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👈 Пожалуйста, загрузите файлы данных в боковой панели для начала анализа.")
    st.stop()

with st.spinner("Обработка данных..."):
    df = process_data(uploaded_files)
    if df is None or df.empty:
        st.error("Не удалось загрузить данные. Проверьте формат файлов.")
        st.stop()
    
    nlp = load_stanza_model()
    unique_axiologemes = df["axiologeme"].unique().tolist()
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = {ax: color_palette[i % len(color_palette)] for i, ax in enumerate(unique_axiologemes)}

# --- ВЕКТОРИЗАЦИЯ (может занять время при первом запуске) ---
with st.spinner("Векторизация контекстов и применение PCA (2D)..."):
    embeddings, coords_2d = get_embeddings_and_pca(df["full_context"].tolist(), n_components=2)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

# --- ВКЛАДКИ ПРИЛОЖЕНИЯ ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Динамика упоминаний", 
    "🗺️ 2D Семантические поля", 
    "🔀 Пересечения полей", 
    "🧲 Внешняя семантическая нагрузка", 
    "🔍 Внутренняя полисемия"
])

# ==============================================================================
# ВКЛАДКА 1: ДИНАМИКА УПОМИНАНИЙ (2D)
# ==============================================================================
with tab1:
    st.header("Диахрония объёма аксиологем")
    
    # Группировка по году и аксиологеме
    if "year" in df.columns:
        volume_time = df.groupby(["year", "axiologeme"]).size().reset_index(name="count")
        
        fig_line = px.line(
            volume_time, x="year", y="count", color="axiologeme", color_discrete_map=colors,
            title="Динамика количества употреблений аксиологем во времени",
            markers=True, line_shape="spline"
        )
        fig_line.update_layout(xaxis_title="Год", yaxis_title="Количество упоминаний", height=500)
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.markdown("### 📊 Статистика по годам")
        st.dataframe(volume_time.pivot(index="year", columns="axiologeme", values="count").fillna(0).astype(int), use_container_width=True)
    else:
        st.warning("Колонка 'year' отсутствует в данных. Невозможно построить временную динамику.")

# ==============================================================================
# ВКЛАДКА 2: 2D СЕМАНТИЧЕСКИЕ ПОЛЯ
# ==============================================================================
with tab2:
    st.header("2D-проекция семантических полей аксиологем")
    st.markdown("Каждая точка представляет собой вектор контекста, спроецированный на 2 главные компоненты (PCA).")
    
    fig_scatter = px.scatter(
        df, x="x", y="y", color="axiologeme", color_discrete_map=colors,
        hover_data=["full_context"], opacity=0.6,
        title="2D-семантическое пространство контекстов"
    )
    fig_scatter.update_layout(xaxis_title="Главная компонента 1 (PC1)", yaxis_title="Главная компонента 2 (PC2)", height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# ВКЛАДКА 3: ПЕРЕСЕЧЕНИЯ ПОЛЕЙ
# ==============================================================================
with tab3:
    st.header("Пересечения лексических и текстовых полей")
    
    top_words = get_top_words(df, nlp, top_n=30)
    
    # 1. Пересечение слов
    word_intersections = []
    for ax1, ax2 in itertools.combinations(top_words.keys(), 2):
        set1, set2 = set(top_words[ax1]), set(top_words[ax2])
        intersect = set1.intersection(set2)
        if intersect:
            word_intersections.append({
                "Аксиологема 1": ax1,
                "Аксиологема 2": ax2,
                "Количество общих слов": len(intersect),
                "Общие слова": ", ".join(list(intersect)[:10]) # Показываем первые 10
            })
    
    st.subheader("Общие слова в топ-30 лексических полей")
    if word_intersections:
        st.dataframe(pd.DataFrame(word_intersections), use_container_width=True)
    else:
        st.info("Пересечений в топ-30 слов не найдено.")

    # 2. Пересечение текстов (упрощённо: косинусное сходство > 0.9)
    st.subheader("Семантически близкие контексты (Cosine Similarity > 0.9)")
    if len(unique_axiologemes) >= 2:
        ax1, ax2 = unique_axiologemes[0], unique_axiologemes[1]
        df1 = df[df["axiologeme"] == ax1].reset_index(drop=True)
        df2 = df[df["axiologeme"] == ax2].reset_index(drop=True)
        
        # Берём случайную подвыборку для ускорения расчёта в Streamlit
        sample1 = df1.sample(min(100, len(df1)))
        sample2 = df2.sample(min(100, len(df2)))
        
        sims = cosine_similarity(sample1["x"].values.reshape(-1, 1), sample2["x"].values.reshape(-1, 1)) # Упрощённо по PC1 для скорости, или использовать полные эмбеддинги
        # Для точности лучше использовать полные эмбеддинги, но ради демо используем 1D проекцию или ограниченный расчёт
        st.write(f"*(Расчёт выполнен на выборке по 100 контекстов для каждой аксиологемы из-за ограничений производительности браузера)*")
        
        # Находим пары с высоким сходством
        high_sim_pairs = []
        emb1 = load_embedding_model().encode(sample1["full_context"].tolist())
        emb2 = load_embedding_model().encode(sample2["full_context"].tolist())
        full_sims = cosine_similarity(emb1, emb2)
        
        for i in range(len(sample1)):
            for j in range(len(sample2)):
                if full_sims[i, j] > 0.85:
                    high_sim_pairs.append({
                        "Сходство": round(full_sims[i, j], 3),
                        f"Контекст ({ax1})": sample1.iloc[i]["full_context"][:100] + "...",
                        f"Контекст ({ax2})": sample2.iloc[j]["full_context"][:100] + "..."
                    })
        
        if high_sim_pairs:
            st.dataframe(pd.DataFrame(high_sim_pairs).head(10), use_container_width=True)
        else:
            st.info("Высокого сходства (>0.85) в выборке не найдено.")

# ==============================================================================
# ВКЛАДКА 4: ВНЕШНЯЯ СЕМАНТИЧЕСКАЯ НАГРУЗКА
# ==============================================================================
with tab4:
    st.header("Внешняя семантическая нагрузка")
    st.markdown("""
    **Метрика:** Отношение количества чужих точек, попавших внутрь 2D-выпуклой оболочки данной аксиологемы, к общему количеству собственных точек этой аксиологемы.
    """)
    
    cluster_points = {ax: df[df["axiologeme"] == ax][["x", "y"]].values for ax in unique_axiologemes}
    external_load = []
    
    for ax in unique_axiologemes:
        hull_pts = cluster_points[ax]
        other_pts = df[df["axiologeme"] != ax][["x", "y"]].values
        
        if len(hull_pts) >= 3:
            inside = points_in_hull_2d(other_pts, hull_pts)
            load_value = inside.sum() / len(hull_pts)
            external_load.append({
                "Аксиологема": ax,
                "Внешняя семантическая нагрузка": round(load_value, 3),
                "Чужих точек внутри оболочки": int(inside.sum()),
                "Собственных точек (размер оболочки)": len(hull_pts)
            })
        else:
            external_load.append({"Аксиологема": ax, "Внешняя семантическая нагрузка": "N/A (мало точек)"})
            
    st.dataframe(pd.DataFrame(external_load), use_container_width=True)

# ==============================================================================
# ВКЛАДКА 5: ВНУТРЕННЯЯ ПОЛИСЕМИЯ
# ==============================================================================
with tab5:
    st.header("Анализ внутренней полисемии (кластеризация)")
    
    target_ax = st.selectbox("Выберите аксиологему для анализа полисемии:", unique_axiologemes)
    max_k = st.slider("Максимальное число кластеров для проверки:", 2, 10, 5)
    
    if st.button("Рассчитать оптимальное число кластеров"):
        df_sub = df[df["axiologeme"] == target_ax].copy()
        
        if len(df_sub) < 10:
            st.warning("Слишком мало данных для кластеризации.")
        else:
            with st.spinner("Кластеризация..."):
                silhouette_scores = {}
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(df_sub[["x", "y"]].values)
                    score = silhouette_score(df_sub[["x", "y"]].values, labels)
                    silhouette_scores[k] = score
                
                optimal_k = max(silhouette_scores, key=silhouette_scores.get)
                
                st.success(f"👉 Оптимальное число подкластеров по метрике Silhouette: **{optimal_k}** (Score: {silhouette_scores[optimal_k]:.3f})")
                
                # Финальная кластеризация
                kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                df_sub["subcluster"] = kmeans_final.fit_predict(df_sub[["x", "y"]].values)
                
                # 2D Визуализация подкластеров
                fig_poly = px.scatter(
                    df_sub, x="x", y="y", color="subcluster", 
                    color_continuous_scale=px.colors.qualitative.Set1,
                    hover_data=["full_context"],
                    title=f"Внутренняя полисемия: «{target_ax}» ({optimal_k} смысловых подтипа)"
                )
                fig_poly.update_layout(height=500)
                st.plotly_chart(fig_poly, use_container_width=True)
                
                # Примеры контекстов
                st.subheader("Примеры контекстов по подкластерам")
                for cluster_id in range(optimal_k):
                    with st.expander(f"Подкластер {cluster_id}"):
                        examples = df_sub[df_sub["subcluster"] == cluster_id]["full_context"].head(3).tolist()
                        for i, ex in enumerate(examples, 1):
                            st.markdown(f"{i}. *{ex}*")

# ==============================================================================
# ДОКУМЕНТАЦИЯ (В БОКОВОЙ ПАНЕЛИ ИЛИ ВНИЗУ)
# ==============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Формулы и методология")
    st.markdown("""
    **1. Векторизация и PCA:**
    Контексты $T_i$ преобразуются в векторы $E_i$ с помощью языковой модели. Метод главных компонент (PCA) снижает размерность до 2D:
    $$X_{2D} = E \cdot W$$
    где $W$ — матрица собственных векторов ковариационной матрицы.

    **2. Центроид (для динамики):**
    Среднее арифметическое координат точек аксиологемы $a$ в год $t$:
    $$C_{a, t} = \\frac{1}{N} \\sum_{i=1}^{N} x_{i}$$

    **3. Выпуклая оболочка (Convex Hull):**
    Минимальный выпуклый многоугольник, содержащий все точки множества. Используется алгоритм Делоне для проверки принадлежности точки $P$ оболочке: $P$ внутри, если она принадлежит одному из симплексов триангуляции.

    **4. Внешняя семантическая нагрузка:**
    $$L_{ext} = \\frac{N_{foreign\_inside}}{N_{own\_total}}$$
    Показывает, насколько семантическое поле одной аксиологемы "поглощает" контексты другой.

    **5. Метрика Silhouette (для полисемии):**
    Оценивает качество кластеризации для точки $i$:
    $$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
    где $a(i)$ — среднее расстояние до точек своего кластера, $b(i)$ — среднее расстояние до точек ближайшего чужого кластера.
    """)
