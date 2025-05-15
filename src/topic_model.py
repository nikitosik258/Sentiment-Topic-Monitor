# topic_model.py
"""
Модуль для тематического моделирования текстов с помощью BERTopic и SentenceTransformer,
с поддержкой использования в общем pipeline и с фиксированным random_state.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from umap import UMAP
import pandas as pd
import plotly.express as px
from umap import UMAP
from hdbscan import HDBSCAN


def get_topics(texts: List[str]) -> Tuple[List[int], BERTopic, List[List[float]]]:
    """
    Обучает BERTopic на входных текстах и возвращает:
    - список тем
    - модель BERTopic
    - эмбеддинги текстов
    """
    if len(texts) < 10:
        raise ValueError("Для тематического моделирования требуется минимум 10 текстов!")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Фиксируем random_state для воспроизводимости
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', prediction_data=True)

    topic_model = BERTopic(language="english", umap_model=umap_model, hdbscan_model=hdbscan_model)

    topics, _ = topic_model.fit_transform(texts, embeddings)
    return topics, topic_model, embeddings


def plot_topics_bar(model: BERTopic):
    """
    Рисует bar-график распределения тем на основе BERTopic модели.
    """
    topics_info = model.get_topic_info()
    topics_info = topics_info[topics_info.Topic != -1]  # исключаем шум

    fig = px.bar(
        topics_info,
        x="Count",
        y="Name",
        orientation="h",
        title="Распределение сообщений по темам",
        labels={"Count": "Количество сообщений", "Name": "Тема"},
        height=600
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.show()