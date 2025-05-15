"""
Модуль визуализации: графики по темам и по тональности.
"""

import pandas as pd
import plotly.express as px
from bertopic import BERTopic


def plot_topic_distribution(model: BERTopic):
    """
    Рисует bar-график распределения сообщений по темам из BERTopic
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


def plot_sentiment_distribution(df: pd.DataFrame):
    """
    Гистограмма распределения тональности (positive/negative)
    """
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    fig = px.bar(
        sentiment_counts,
        x="sentiment",
        y="count",
        title="Распределение тональности отзывов",
        labels={"sentiment": "Тональность", "count": "Количество"},
        color="sentiment"
    )
    fig.show()


# Пример использования:
# plot_sentiment_distribution(df)
# plot_topic_distribution(model)