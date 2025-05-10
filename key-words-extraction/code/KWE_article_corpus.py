#!/usr/bin/env python3
"""
Извлечение ключевых выражений из локального корпуса статей с помощью RuTerm и YAKE
результат сохраняется в общий CSV-файл с полями:
Index, Year, Name, RuTerm, RuTerm_filtered, YAKE
"""
import os
import re
import pandas as pd
import yake
from rutermextract import TermExtractor
from pathlib import Path

# Путь к локальной папке корпуса
BASE_FOLDER = '/Users/juliak/Downloads/IMS2013-20242'

# Настройки
TOP_N = 10
# Папка для сохранения результата
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'keywords_corpus.csv'

# Фильтр для термов RuTerm
def is_valid_term(term: str) -> bool:
    words = term.split()
    return all(len(w) >= 4 and re.fullmatch(r'[а-яА-ЯёЁ-]+', w) for w in words)

# Извлечение RuTerm (базовое)
def extract_ruterm(text: str, top_n: int = TOP_N) -> str:
    extractor = TermExtractor()
    try:
        terms = extractor(text)
        keywords = [t.normalized for t in terms[:top_n]]
    except Exception:
        keywords = []
    return ', '.join(keywords)

# Извлечение RuTerm с фильтрами
def extract_ruterm_filtered(text: str, top_n: int = TOP_N) -> str:
    extractor = TermExtractor()
    try:
        terms = extractor(text)
        filtered = [t.normalized for t in terms if is_valid_term(t.normalized)]
        seen = set()
        uniq = []
        for term in filtered:
            if term not in seen:
                seen.add(term)
                uniq.append(term)
            if len(uniq) >= top_n:
                break
        keywords = uniq
    except Exception:
        keywords = []
    return ', '.join(keywords)

# Извлечение YAKE (uni + bi граммы)
def extract_yake(text: str, top_n: int = TOP_N) -> str:
    kw1 = yake.KeywordExtractor(lan="ru", n=1, top=top_n).extract_keywords(text)
    kw2 = yake.KeywordExtractor(lan="ru", n=2, top=top_n).extract_keywords(text)
    unique = {}
    for kw, score in kw1 + kw2:
        if kw not in unique or score < unique[kw]:
            unique[kw] = score
    sorted_kw = sorted(unique.items(), key=lambda x: x[1])[:top_n]
    return ', '.join(kw for kw, _ in sorted_kw)


def main():
    results = []
    idx = 1
    for year_folder in sorted(os.listdir(BASE_FOLDER)):
        year_path = os.path.join(BASE_FOLDER, year_folder)
        if not os.path.isdir(year_path):
            continue
        for filename in sorted(os.listdir(year_path)):
            if not filename.endswith('.txt') or 'Abstract' in filename or 'KW' in filename:
                continue
            parts = filename.split('_')
            try:
                ims_i = parts.index('IMS')
                author = '_'.join(parts[:ims_i])
                year = ''.join(filter(str.isdigit, parts[ims_i+1]))
            except Exception:
                continue
            file_path = os.path.join(year_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                text = ''
            kw_rt   = extract_ruterm(text)
            kw_rt_f = extract_ruterm_filtered(text)
            kw_yake = extract_yake(text)
            results.append({
                'Index': idx,
                'Year': year,
                'Name': author,
                'RuTerm': kw_rt,
                'RuTerm_filtered': kw_rt_f,
                'YAKE': kw_yake
            })
            idx += 1
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved combined keywords to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
