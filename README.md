# 🧠 Sentiment & Topic Monitor

Проект по автоматическому анализу отзывов и текстов: определение **тональности** (позитив/негатив), выделение **тематики сообщений** и визуализация результатов.

## 🔍 Что делает этот проект

- Загружает и обрабатывает текстовые данные (отзывы, обращения, комментарии)
- Классифицирует тексты по **тональности** с помощью `DistilBERT`
- Выделяет скрытые **темы** в отзывах с помощью `BERTopic`
- Строит **интерактивные графики**:
  - Распределение тональности
  - Частота появления тем
- Сохраняет результаты анализа в CSV

---

## 🏗 Стек технологий

- **Python 3.10+**
- `pandas`, `nltk`, `plotly`, `scikit-learn`
- `transformers` (HuggingFace)
- `sentence-transformers` (эмбеддинги)
- `bertopic` (тематическое моделирование)
- `UMAP`, `HDBSCAN` (кластеризация тем)

---

## 📦 Установка

```bash
git clone https://github.com/yourusername/sentiment-topic-monitor.git
cd sentiment-topic-monitor

pip install -r requirements.txt

# Также нужно один раз скачать nltk stopwords
python -c "import nltk; nltk.download('stopwords')"
