"""
Главный модуль, объединяющий все этапы анализа:
1. Загрузка данных из fastText-формата
2. Предобработка текста
3. Анализ тональности
4. Тематическое моделирование
5. Визуализация
"""

from load_data import load_fasttext_format
from preprocessing import preprocess_texts
from sentiment import analyze_sentiment
from topic_model import get_topics
from viz_utils import plot_sentiment_distribution, plot_topic_distribution

import pandas as pd

def run_pipeline(data_path: str, limit: int = 5000):
    # 1. Загрузка данных
    df = load_fasttext_format(data_path)
    print(f"Загружено {len(df)} записей")

    # Ограничение объема выборки
    df = df.head(limit)
    print(f"Анализируем первые {len(df)} записей")

    # 2. Предобработка текста
    print("Предобработка...")
    df["clean_text"] = preprocess_texts(df["text"].tolist())

    # 3. Анализ тональности
    print("Анализ тональности...")
    sentiments, confidences = analyze_sentiment(df["text"].tolist())
    df["sentiment"] = sentiments
    df["sentiment_confidence"] = confidences

    # 4. Тематическое моделирование
    print("Темы...")
    topics, model, _ = get_topics(df["clean_text"].tolist())
    df["topic"] = topics

    # 5. Визуализация
    print("Визуализация...")
    plot_sentiment_distribution(df)
    plot_topic_distribution(model)

    # Сохранение результатов
    df.to_csv("monitoring_results.csv", index=False)
    print("Результаты сохранены в monitoring_results.csv")


if __name__ == "__main__":
    run_pipeline("data/train.ft.txt", limit=5000)