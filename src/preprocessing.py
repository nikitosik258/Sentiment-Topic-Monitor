"""
Модуль предобработки текста: очистка, токенизация, стемминг, удаление стоп-слов (на базе nltk).
"""

import re
from typing import List
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Инициализация компонентов
tokenizer = RegexpTokenizer(r"\b\w+\b")
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def clean_text(text: str) -> str:
    """
    Очистка текста от html, спецсимволов, приведение к нижнему регистру.
    """
    text = re.sub(r"<.*?>", "", text)  # удаление html
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # удаление пунктуации
    return text.lower().strip()

def tokenize_stem(text: str) -> str:
    """
    Токенизация, удаление стоп-слов и стемминг
    """
    tokens = tokenizer.tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Полный пайплайн предобработки: очистка → токенизация → удаление стоп-слов → стемминг
    """
    return [tokenize_stem(clean_text(t)) for t in texts]

# Пример использования:
# import nltk
# nltk.download('stopwords')
# cleaned = preprocess_texts(["This is <b>awesome</b>!!! but... expensive."])
# print(cleaned)  # ['awesom expens']
