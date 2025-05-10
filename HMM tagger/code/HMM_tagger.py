"""
Реализация POS-теггинга на основе скрытой марковской модели
с обучением на размеченном корпусе "train.txt" и теггингом "test.txt".
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
PARSE_DIR = BASE_DIR / 'parse'
OUTPUT_DIR = BASE_DIR / 'output'
TRAIN_FILE = PARSE_DIR / 'GSD_train.txt'
TEST_FILE  = PARSE_DIR / 'GSD_test.txt'

SMALL_PROB = 1e-8
SEED = 1234


def load_sentences(path: Path):
    """
    Загружает размеченные предложения из файла.
    """
    text = path.read_text(encoding='utf-8-sig')
    blocks = [b for b in text.strip().split('\n\n') if b]
    sents = []
    for block in blocks:
        sent = []
        for line in block.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                word, tag = parts[1], parts[3]
                sent.append((word, tag))
        if sent:
            sents.append(sent)
    return sents


def prepare_data():
    """
    Загружает train и test, объединяет их, и разбивает на train/test 80/20.
    Возвращает train_set, test_set, train_bag, test_base, test_words.
    """
    all_sents = load_sentences(TRAIN_FILE) + load_sentences(TEST_FILE)
    train_set, test_set = train_test_split(all_sents, train_size=0.8, random_state=SEED)
    train_bag  = [pair for sent in train_set for pair in sent]
    test_base  = [pair for sent in test_set  for pair in sent]
    test_words = [w for (w,_) in test_base]
    return train_set, test_set, train_bag, test_base, test_words


def train_counts(train_set, train_bag):
    """
    Вычисляет счётчики начальных тегов, переходов и эмиссий.
    Возвращает init_counts, tag_counts, emit_counts, trans_counts.
    """
    from collections import Counter, defaultdict
    init_counts = Counter()
    tag_counts = Counter()
    emit_counts = defaultdict(Counter)
    trans_counts = defaultdict(Counter)

    for sent in train_set:
        first_tag = sent[0][1]
        init_counts[first_tag] += 1
        tag_counts[first_tag] += 1
        emit_counts[first_tag][sent[0][0]] += 1
        prev_tag = first_tag
        for word, tag in sent[1:]:
            tag_counts[tag] += 1
            emit_counts[tag][word] += 1
            trans_counts[prev_tag][tag] += 1
            prev_tag = tag
    return init_counts, tag_counts, emit_counts, trans_counts


def train_probs(init_counts, tag_counts, emit_counts, trans_counts):
    """
    Вычисляет вероятности pi, A и B из счётчиков.
    pi[tag] = P(tag_0)
    A[t1][t2] = P(t2|t1)
    B[tag][word] = P(word|tag)
    """
    total_init = sum(init_counts.values())
    pi = {tag: init_counts[tag]/total_init for tag in init_counts}

    A = {}
    for t1, nxt in trans_counts.items():
        total = sum(nxt.values())
        A[t1] = {t2: cnt/total for t2, cnt in nxt.items()}

    B = {}
    for tag, wc in emit_counts.items():
        total = tag_counts[tag]
        B[tag] = {w: cnt/total for w, cnt in wc.items()}

    return pi, A, B


def viterbi_fast(words, tags, pi, A, B):
    """
    Оптимизированный алгоритм Витерби, не использует логарифмы.
    """
    state = []
    for idx, w in enumerate(words):
        best_tag, best_score = None, 0.0
        for tag in tags:
            if idx == 0:
                trans_p = pi.get(tag, SMALL_PROB)
            else:
                trans_p = A.get(state[-1], {}).get(tag, SMALL_PROB)
            emis_p = B.get(tag, {}).get(w, SMALL_PROB)
            score = trans_p * emis_p
            if score > best_score:
                best_score, best_tag = score, tag
        state.append(best_tag)
    return list(zip(words, state))


def evaluate(test_words, test_base, tags, pi, A, B):
    """
    Применяет Viterbi, вычисляет точность и список mismatches.
    """
    tagged = viterbi_fast(test_words, tags, pi, A, B)
    correct = sum(1 for (w,p),(w2,gt) in zip(tagged, test_base) if p==gt)
    total = len(test_base)
    acc = correct/total if total else 0.0
    mismatches = [(w,p,gt) for (w,p),(w2,gt) in zip(tagged, test_base) if p!=gt]
    return acc, mismatches, tagged


def save_results(pi, A, B, acc, mismatches):
    """
    Сохраняет transition_probs, emission_probs, accuracy и mismatches.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A).fillna(0).to_csv(OUTPUT_DIR / 'transition_probs.csv')
    pd.DataFrame(B).fillna(0).to_csv(OUTPUT_DIR / 'emission_probs.csv')
    with open(OUTPUT_DIR / 'predicted_tags.txt', 'w', encoding='utf-8') as f:
        # можно дополнительно сохранять predictions
        pass
    pd.DataFrame([{'accuracy': acc}]).to_csv(OUTPUT_DIR / 'accuracy.csv', index=False)
    pd.DataFrame(mismatches, columns=['word','predicted','gold']).to_csv(
        OUTPUT_DIR / 'mismatches.csv', index=False)


def main():
    random.seed(SEED)
    train_set, test_set, train_bag, test_base, test_words = prepare_data()
    init_c, tag_c, emit_c, trans_c = train_counts(train_set, train_bag)
    pi, A, B = train_probs(init_c, tag_c, emit_c, trans_c)
    tags = sorted(tag_c)
    acc, mismatches, _ = evaluate(test_words, test_base, tags, pi, A, B)
    print(f"Accuracy: {acc*100:.2f}%")
    save_results(pi, A, B, acc, mismatches)


if __name__ == '__main__':
    main()


