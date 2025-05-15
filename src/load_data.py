# load_data.py
"""
Модуль загрузки данных из fastText-формата с Amazon Reviews.
Ожидается, что каждая строка начинается с __label__1 или __label__2, далее идёт текст отзыва.
__label__1 — негатив
__label__2 — позитив
"""

import pandas as pd

def load_fasttext_format(path: str) -> pd.DataFrame:
    """
    Загружает датасет fastText-формата и возвращает DataFrame с колонками:
    - sentiment (positive/negative)
    - text (str)
    """
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if not line.startswith('__label__'):
                continue  # Пропускаем строки без метки
            try:
                label, text = line.split(' ', 1)
                sentiment = 'positive' if label == '__label__2' else 'negative'
                data.append((sentiment, text))
            except ValueError:
                continue  # Строка не разбивается корректно

    df = pd.DataFrame(data, columns=['sentiment', 'text'])
    return df

# Пример использования:
#df = load_fasttext_format("data/train.ft.txt")
#print(df.head(11))
