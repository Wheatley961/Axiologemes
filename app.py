import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import itertools
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import stanza
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# НАСТРОЙКА СТРАНИЦЫ И КЭШИРОВАНИЕ
# ==============================================================================
st.set_page_config(page_title="Анализ аксиологем", layout="wide", page_icon="📊")

@st.cache_resource
def load_stanza():
    """Загрузка модели лемматизации Stanza (кэшируется)"""
    stanza.download('ru', verbose=False)
    return stanza.Pipeline('ru', processors='tokenize,lemma', verbose=False)

@st.cache_resource
def load_embedding_model():
    """Загрузка модели векторизации (кэшируется)"""
    return SentenceTransformer('cointegrated/rubert-tiny2')

nlp = load_stanza()
model = load_embedding_model()

def lemmatize_text(text):
    """Лемматизация текста с помощью Stanza"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    doc = nlp(text)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma is not None]
    return " ".join(lemmas)

def points_in_hull(points, hull_points):
    """Проверяет, какие точки лежат внутри выпуклой оболочки (Delaunay)"""
    if len(hull_points) < 4:
        return np.array([False] * len(points))
    try:
        hull = Delaunay(hull_points)
        return hull.find_simplex(points) >= 0
    except Exception:
        return np.array([False] * len(points))

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА ДАННЫХ
# ==============================================================================
st.sidebar.header("📂 Загрузка данных")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите файлы Excel (например, ruscorpora_content_здоровье.xlsx)", 
    type="xlsx", 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👈 Пожалуйста, загрузите один или несколько файлов Excel для начала анализа.")
    st.stop()

with st.spinner("Загрузка и предобработка данных..."):
    dfs = []
    for file in uploaded_files:
        filename = file.name
        # Извлечение имени аксиологемы из имени файла
        axiologeme = filename.replace("ruscorpora_content_", "").replace(".xlsx", "")
        
        df_tmp = pd.read_excel(file)
        df_tmp["axiologeme"] = axiologeme
        dfs.append(df_tmp)
    
    df = pd.concat(dfs, ignore_index=True).fillna("")
    
    # Формирование полного контекста
    df["full_context"] = (
        df["Left context"].astype(str) + " " + 
        df["Center"].astype(str) + 
        df["Punct"].astype(str) + " " + 
        df["Right context"].astype(str)
    )
    df["full_context"] = df["full_context"].str.replace(r"\s+", " ", regex=True)
    
    # Лемматизация (с прогресс-баром для больших данных)
    if "lemmatized_context" not in df.columns:
        with st.spinner("Лемматизация текстов (может занять время)..."):
            df["lemmatized_context"] = df["full_context"].apply(lemmatize_text)
    
    # Векторизация и PCA
    with st.spinner("Векторизация и снижение размерности (PCA)..."):
        texts_to_encode = df["lemmatized_context"].tolist()
        embeddings = model.encode(texts_to_encode, show_progress_bar=False)
        
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        df["x"] = embeddings_3d[:, 0]
        df["y"] = embeddings_3d[:, 1]
        df["z"] = embeddings_3d[:, 2]

    st.success(f"✅ Загружено {len(df)} контекстов. Аксиологемы: {', '.join(df['axiologeme'].unique())}")

# ==============================================================================
# РАСЧЁТ МЕТРИК И АНАЛИЗ
# ==============================================================================
cluster_points = {
    ax: df[df["axiologeme"] == ax][["x", "y", "z"]].values
    for ax in df["axiologeme"].unique()
}

color_palette = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
colors = {ax: color for ax, color in zip(df["axiologeme"].unique(), itertools.cycle(color_palette))}

# 1. Расчёт проницаемости и внешней нагрузки
permeability_data = []
external_load_data = []
typology_rows = []

for ax in df["axiologeme"].unique():
    hull_points = cluster_points[ax]
    other_points = df[df["axiologeme"] != ax][["x", "y", "z"]].values
    
    # Проницаемость: доля чужих текстов, попавших в оболочку текущей аксиологемы
    inside = points_in_hull(other_points, hull_points)
    perm_value = inside.sum() / len(other_points) if len(other_points) > 0 else 0.0
    
    # Внешняя нагрузка: число чужих текстов внутри оболочки, нормированное по собственному объёму
    load_value = inside.sum() / len(hull_points) if len(hull_points) > 0 else 0.0
    
    permeability_data.append({"axiologeme": ax, "text_permeability": round(perm_value, 3)})
    external_load_data.append({"axiologeme": ax, "external_semantic_load": round(load_value, 3)})
    
    # Плотность кластера (среднее евклидово расстояние до центроида)
    centroid = hull_points.mean(axis=0)
    density = np.mean([np.linalg.norm(p - centroid) for p in hull_points])
    
    # Уровень проницаемости (категориальный)
    if perm_value < 0.05:
        perm_level = "низкая"
    elif perm_value < 0.2:
        perm_level = "умеренная"
    else:
        perm_level = "высокая"
        
    typology_rows.append({
        "axiologeme": ax,
        "volume": len(hull_points),
        "density": round(density, 3),
        "permeability": round(perm_value, 3),
        "permeability_level": perm_level,
        "external_semantic_load": round(load_value, 3)
    })

permeability_df = pd.DataFrame(permeability_data)
external_load_df = pd.DataFrame(external_load_data)
typology_df = pd.DataFrame(typology_rows).sort_values("volume", ascending=False)

# 2. Анализ пересечений текстовых кластеров
text_intersections = []
for ax1, ax2 in itertools.combinations(df["axiologeme"].unique(), 2):
    points1 = df[df["axiologeme"] == ax1][["x", "y", "z"]].values
    points2 = df[df["axiologeme"] == ax2][["x", "y", "z"]].values
    
    in_hull2 = points_in_hull(points1, points2)
    in_hull1 = points_in_hull(points2, points1)
    
    total_intersect = in_hull2.sum() + in_hull1.sum()
    ratio = round(total_intersect / (len(points1) + len(points2)), 3) if (len(points1) + len(points2)) > 0 else 0.0
    
    text_intersections.append({
        "axiologeme_1": ax1,
        "axiologeme_2": ax2,
        "intersection_count": int(total_intersect),
        "ratio": ratio
    })
text_intersections_df = pd.DataFrame(text_intersections).sort_values("intersection_count", ascending=False)

# 3. Анализ лексических полей (Топ-50 слов)
top_words = {}
all_lemmas = df["lemmatized_context"].str.split().explode().value_counts()
vocab = all_lemmas.head(5000).index.tolist()
vocab_vectors = model.encode(vocab)

for ax in df["axiologeme"].unique():
    center_forms = df[df["axiologeme"] == ax]["Center"].unique()
    center_lemmas = [lemmatize_text(w) for w in center_forms]
    center_vecs = model.encode(center_lemmas)
    ax_centroid_vec = center_vecs.mean(axis=0)
    
    similarities = cosine_similarity([ax_centroid_vec], vocab_vectors)[0]
    top_idx = similarities.argsort()[-50:][::-1]
    top_words[ax] = [vocab[i] for i in top_idx]

# ==============================================================================
# ИНТЕРФЕЙС STREAMLIT: ВКЛАДКИ
# ==============================================================================
st.title("📊 Комплексный анализ аксиологем")
st.markdown("Анализ семантических пространств, проницаемости, внешней нагрузки и лексических полей на основе корпусных данных.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Типология и Метрики", 
    "🌐 3D Визуализация контекстов", 
    "🔤 Лексические поля", 
    "🧩 Внутренняя полисемия", 
    "📚 Документация и Формулы"
])

# --- ВКЛАДКА 1: Типология и Метрики ---
with tab1:
    st.subheader("Комплексная типология аксиологем")
    st.dataframe(typology_df, use_container_width=True)
    
    st.subheader("Матрица пересечений текстовых кластеров")
    st.dataframe(text_intersections_df, use_container_width=True)

# --- ВКЛАДКА 2: 3D Визуализация ---
with tab2:
    st.subheader("3D-семантическое пространство контекстов с границами кластеров")
    fig_3d = go.Figure()
    
    for ax in df["axiologeme"].unique():
        subset = df[df["axiologeme"] == ax]
        hover_texts = [
            f"<b>Аксиологема:</b> {ax}<br><b>Центр:</b> {c}<br><b>Контекст:</b> {ctx}"
            for c, ctx in zip(subset["Center"], subset["full_context"])
        ]
        
        # Точки
        fig_3d.add_trace(go.Scatter3d(
            x=subset["x"], y=subset["y"], z=subset["z"],
            mode="markers",
            marker=dict(size=3, color=colors[ax]),
            text=hover_texts,
            hoverinfo="text",
            name=f"точки: {ax}"
        ))
        
        # Выпуклая оболочка
        points = cluster_points[ax]
        if len(points) >= 4:
            try:
                hull = Delaunay(points)
                fig_3d.add_trace(go.Mesh3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                    opacity=0.15, color=colors[ax],
                    name=f"границы: {ax}"
                ))
            except Exception:
                pass

    fig_3d.update_layout(
        title="3D-семантическое пространство контекстов",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=800, height=700,
        legend=dict(title="Аксиологемы")
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("Диахрония объёма аксиологем")
    # Преобразование даты и группировка по годам (без анимации дрейфа, как и просили)
    df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
    df["year"] = df["Created"].dt.year
    volume_time = df.groupby(["axiologeme", "year"]).size().reset_index(name="count")
    
    fig_line = px.line(
        volume_time, x="year", y="count", color="axiologeme",
        title="Динамика объёма упоминаний аксиологем по годам", markers=True
    )
    fig_line.update_layout(xaxis_title="Год", yaxis_title="Количество употреблений")
    st.plotly_chart(fig_line, use_container_width=True)

# --- ВКЛАДКА 3: Лексические поля ---
with tab3:
    st.subheader("3D-визуализация лексических полей аксиологем")
    fig_words = go.Figure()
    
    for ax, words in top_words.items():
        word_vecs = model.encode(words)
        word_vecs_3d = pca.transform(word_vecs)
        
        hover_words = [f"<b>Аксиологема:</b> {ax}<br><b>Слово:</b> {w}" for w in words]
        
        fig_words.add_trace(go.Scatter3d(
            x=word_vecs_3d[:, 0], y=word_vecs_3d[:, 1], z=word_vecs_3d[:, 2],
            mode="markers+text",
            text=words,
            textposition="top center",
            marker=dict(size=5, color=colors[ax]),
            textfont=dict(size=10, color="black"),
            hoverinfo="text",
            hovertext=hover_words,
            name=ax
        ))
        
    fig_words.update_layout(
        title="3D-семантические поля аксиологем (Топ-50 слов)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=800, height=700
    )
    st.plotly_chart(fig_words, use_container_width=True)

# --- ВКЛАДКА 4: Внутренняя полисемия ---
with tab4:
    st.subheader("Автоматический анализ внутренней полисемии аксиологемы")
    target_ax = st.selectbox("Выберите аксиологему для кластеризации:", list(df["axiologeme"].unique()))
    
    if st.button("Запустить анализ полисемии"):
        df_sub = df[df["axiologeme"] == target_ax].copy()
        X = df_sub[["x", "y", "z"]].values
        
        silhouette_scores = {}
        for k in range(2, 6):
            if len(X) <= k:
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores[k] = score
            
        if silhouette_scores:
            optimal_k = max(silhouette_scores, key=silhouette_scores.get)
            st.success(f"✅ Оптимальное число подкластеров (по силуэту): **{optimal_k}** (Score: {silhouette_scores[optimal_k]:.3f})")
            
            # Пересчёт с оптимальным K
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df_sub["subcluster"] = kmeans_final.fit_predict(X)
            
            # Визуализация подкластеров
            fig_poly = go.Figure()
            for cluster_id in range(optimal_k):
                subset_c = df_sub[df_sub["subcluster"] == cluster_id]
                fig_poly.add_trace(go.Scatter3d(
                    x=subset_c["x"], y=subset_c["y"], z=subset_c["z"],
                    mode="markers",
                    marker=dict(size=4, color=colors[target_ax], opacity=0.6),
                    name=f"Подкластер {cluster_id}"
                ))
            
            fig_poly.update_layout(
                title=f"Внутренняя полисемия: «{target_ax}» (K={optimal_k})",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                height=600
            )
            st.plotly_chart(fig_poly, use_container_width=True)
            
            # Примеры контекстов
            st.markdown("#### Примеры контекстов по смысловым подтипам:")
            for cluster_id in range(optimal_k):
                with st.expander(f"📂 Подкластер {cluster_id}"):
                    examples = df_sub[df_sub["subcluster"] == cluster_id].head(5)["full_context"].values
                    for i, ctx in enumerate(examples, 1):
                        st.markdown(f"{i}. *{ctx}*")
        else:
            st.warning("Недостаточно данных для кластеризации (требуется > 5 контекстов).")

# --- ВКЛАДКА 5: Документация и Формулы ---
with tab5:
    st.header("📚 Математические основы и описание метрик")
    st.markdown("""
    Данный анализ опирается на представление текстовых контекстов в виде точек в трёхмерном семантическом пространстве (после применения PCA к эмбеддингам `rubert-tiny2`). 
    Ниже приведены формулы для расчёта ключевых метрик.
    """)
    
    st.subheader("1. Плотность кластера (Density)")
    st.markdown(r"""
    Характеризует компактность семантического поля аксиологемы. Рассчитывается как среднее евклидово расстояние от всех точек кластера до его центроида.
    
    $$ D = \frac{1}{N} \sum_{i=1}^{N} \| p_i - C \| $$
    
    Где:
    - $N$ — общее количество текстовых контекстов данной аксиологемы.
    - $p_i$ — координаты $i$-го контекста в 3D-пространстве $(x_i, y_i, z_i)$.
    - $C$ — координаты центроида кластера (среднее арифметическое всех точек).
    - $\| \cdot \|$ — евклидова норма (расстояние).
    """)
    
    st.subheader("2. Семантическая проницаемость (Text Permeability)")
    st.markdown(r"""
    Показывает, насколько "проницаема" семантическая оболочка аксиологемы для чужих контекстов. 
    Рассчитывается как доля текстов **других** аксиологем, которые попали внутрь выпуклой оболочки (Convex Hull) текущей аксиологемы.
    
    $$ P = \frac{| T_{other} \cap Hull_{current} |}{| T_{other} |} $$
    
    Где:
    - $T_{other}$ — множество всех текстовых точек, не принадлежащих текущей аксиологеме.
    - $Hull_{current}$ — выпуклая оболочка, построенная по точкам текущей аксиологемы (алгоритм триангуляции Делоне).
    - $| \cdot |$ — мощность множества (количество точек).
    
    *Интерпретация*: Высокая проницаемость (> 0.2) указывает на то, что семантическое поле аксиологемы широко и включает в себя много чужеродных контекстов (размытые границы).
    """)
    
    st.subheader("3. Внешняя семантическая нагрузка (External Semantic Load)")
    st.markdown(r"""
    Показывает, насколько сильно текущая аксиологема "поглощает" чужие смыслы относительно своего собственного объёма.
    
    $$ L = \frac{| T_{other} \cap Hull_{current} |}{| T_{current} |} $$
    
    Где:
    - $T_{current}$ — множество собственных текстовых точек аксиологемы.
    - Числитель тот же, что и в формуле проницаемости.
    
    *Интерпретация*: Если $L > 1$, это означает, что внутрь оболочки данной аксиологемы попало больше чужих текстов, чем в ней содержится своих собственных.
    """)
    
    st.subheader("4. Коэффициент пересечения кластеров (Intersection Ratio)")
    st.markdown(r"""
    Оценивает взаимное перекрытие двух аксиологем в обе стороны.
    
    $$ R = \frac{| T_1 \cap Hull_2 | + | T_2 \cap Hull_1 |}{| T_1 | + | T_2 |} $$
    
    Где:
    - $T_1, T_2$ — множества точек первой и второй аксиологем соответственно.
    """)
    
    st.subheader("5. Векторизация и снижение размерности")
    st.markdown(r"""
    1. **Лемматизация**: Приведение слов к начальной форме с помощью `Stanza`.
    2. **Эмбеддинг**: Преобразование текста в плотный вектор $v \in \mathbb{R}^{d}$ с помощью модели `cointegrated/rubert-tiny2`.
    3. **PCA (Principal Component Analysis)**: Линейное преобразование для проецирования векторов в 3D-пространство ($d \rightarrow 3$) с сохранением максимальной дисперсии данных.
       $$ X_{3D} = X_{orig} \cdot W $$
       Где $W$ — матрица собственных векторов ковариационной матрицы данных.
    """)
