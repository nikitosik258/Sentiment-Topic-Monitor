"""
Модуль для анализа тональности текстов с помощью модели от HuggingFace.
Метки: positive / negative
"""

from transformers import pipeline
from typing import List, Tuple

# Загружаем пайплайн анализа тональности (по умолчанию — английский DistilBERT)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(texts: List[str]) -> Tuple[List[str], List[float]]:
    """
    Анализ тональности списка текстов.

    Возвращает:
    - список меток (positive / negative)
    - список confidence-оценок (float)
    """
    labels = []
    scores = []
    results = sentiment_pipeline(texts, truncation=True)
    for r in results:
        labels.append("positive" if r["label"] == "POSITIVE" else "negative")
        scores.append(round(r["score"], 4))
    return labels, scores

# Пример использования:
labels, confs = analyze_sentiment(["I love it!", "This sucks."])
print(list(zip(labels, confs)))