import sys
from pathlib import Path
import re
from collections import defaultdict
from typing import List, Dict, Set

# 1. Предобработка текста

# Загрузка списка стоп-слов из текстового файла
def load_stopwords(filepath='stopwords.txt'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Ошибка: Файл стоп-слов '{filepath}' не найден.")
        return set()

# Очистка текста и извлечения значимых слов
def tokenize_and_filter(text: str, stop_words: Set[str]):
    lower_text = text.lower()
    # все слова, состоящие из русских или английских букв
    raw_tokens = re.findall(r'[a-zа-яё]+', lower_text)
    # Фильтруем: убираем стоп-слова и слишком короткие слова
    filtered_tokens = [word for word in raw_tokens if word not in stop_words and len(word) >= 2]
    return filtered_tokens

# 2. Построение взвешенного графа
def build_graph(tokens: List[str], window: int = 2):
    graph = defaultdict(lambda: defaultdict(float))
    # перебираем возможные пары, в которых расстояние между словами не превышает размер окна (скольязящее окно)
    for i, word1 in enumerate(tokens):
        for j in range(i+1, min(i+window, len(tokens))):
            word2 = tokens[j]
            if word1 != word2:
                # Увеличиваем вес связи для word1->word2 и для word2->word1
                graph[word1][word2] += 1.0
                graph[word2][word1] += 1.0
    # Удаляем возможные петли, если слово вдруг соединено с самим собой.
    for node in graph:
        if node in graph[node]:
            del graph[node][node]
    return graph


# 3. Итеративный расчёт рангов (алгоритм взвешенного PageRank) 

def textrank_keywords(
    graph: Dict[str, Dict[str, float]],
    d: float = 0.85,
    eps: float = 1e-6,
    max_iter: int = 200
):
    
    # Инициализация рангов всех вершин значением 1.0
    rank = {node: 1.0 for node in graph}
    if not graph:
        return rank

    # Предварительный расчёт сумм весов для каждой вершины (используется в знаменателе формулы TextRank)
    out_sum = {}
    for node in graph:
        # Суммируем все веса рёбер исходящих из node
        total = sum(graph[node].values())
        # Для изолированных вершин (total = 0) ставим 1, чтобы не делить на ноль
        out_sum[node] = total if total > 0 else 1.0

    # Основной цикл с проверкой сходимости

    for _ in range(max_iter):
        new_rank = {}
        max_delta = 0.0

        # Пересчёт ранга для каждого слова
        for node in graph:
            total_incoming = 0.0
            for neighbor in graph[node]:
                # graph[neighbor] — это словарь,
                # graph[neighbor][node] — это вес ребра между neighbour и node.
                weight = graph[neighbor][node]
                # Доля веса, которую neighbour передаёт node
                total_incoming += (weight / out_sum[neighbor]) * rank[neighbor]
            
            # Основная формула пересчёта
            new_rank[node] = (1 - d) + d * total_incoming
            
            # Отслеживаем максимальное абсолютное изменение ранга (Проверка сходимости)
            delta = abs(new_rank[node] - rank[node])
            if delta > max_delta:
                max_delta = delta

        # Обновляем ранги
        rank = new_rank
        
        # Критерий остановки: если максимальное изменение меньше заданной точности eps
        if max_delta < eps:
            break

    return rank

# 4. Извлечение топ-N ключевых слов
def extract_top_keywords(rank: Dict[str, float], top_n: int = 10):
    sorted_keywords = sorted(rank.items(), key=lambda item: item[1], reverse=True)
    return sorted_keywords[:top_n]

# 5. Главная функция
def textrank_pipeline(text: str, stop_words_set: Set[str], window: int = 2, damping: float = 0.85, top_n: int = 10):
    tokens = tokenize_and_filter(text, stop_words_set)
    if len(tokens) < 2:
        return []
    graph = build_graph(tokens, window)
    ranks = textrank_keywords(graph, d=damping)
    top_keywords = extract_top_keywords(ranks, top_n)
    return top_keywords

if __name__ == "__main__":
    stopwords_file = "stopwords_big.txt"
    stop_words_set = load_stopwords(stopwords_file)

    # Аннотации книг
    annotations_dir = Path("book_annotations")
    annotations = {}
    if annotations_dir.exists() and annotations_dir.is_dir():
        for filepath in annotations_dir.glob("*.txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    annotations[filepath.stem] = f.read()
            except Exception as e:
                print(f"Не удалось прочитать {filepath.name}: {e}")
    else:
        print(f"Папка '{annotations_dir}' не найдена.")

    if not annotations:
        print("Папка с аннотациями не найдена или пуста.")


    # Запуск обработки для всех аннотаций
    print("Ключевые слова для книжных аннотаций\n")
    for book_name, book_text in annotations.items():
        keywords = textrank_pipeline(book_text, stop_words_set, top_n=50)
        print(f"**{book_name}**")
        if keywords:
            print("Ключевые слова и их релевантность (ранг):")
            for word, rank_val in keywords:
                print(f"   {word}: {rank_val:.4f}")
        else:
            print("Не удалось извлечь ключевые слова (текст слишком короткий).")
        print("-" * 40)
        