"""
Сравнение двух популярных методов автоматической идентификации языка текста
(langdetect и langid) на наборе примеров русского, английского и немецкого текстов.
"""
from pathlib import Path
import csv
import pandas as pd
from langdetect import detect, DetectorFactory
import langid

# Зафиксируем случайность для langdetect
DetectorFactory.seed = 0

# Пути к папкам и файлам
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'parse'
OUTPUT_FILE = BASE_DIR / 'output' / 'results.csv'

def load_text(file_path: Path) -> str:
    """Загружает текст из файла и возвращает его содержимое."""
    return file_path.read_text(encoding='utf-8-sig')


def classify_langdetect(text: str) -> str:
    """Определяет язык с помощью langdetect."""
    try:
        return detect(text)
    except Exception:
        return 'unknown'


def classify_langid(text: str) -> (str, float):
    """Определяет язык и уверенность с помощью langid."""
    lang, score = langid.classify(text)
    return lang, score


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # Проходим по всем .txt файлам в папке parse (на уровне проекта)
    for txt_file in sorted(INPUT_DIR.glob('*.txt')):
        text = load_text(txt_file)
        # langdetect
        ld = classify_langdetect(text)
        rows.append({
            'file': txt_file.name,
            'method': 'langdetect',
            'predicted': ld,
            'score': ''
        })
        # langid
        li, score = classify_langid(text)
        rows.append({
            'file': txt_file.name,
            'method': 'langid',
            'predicted': li,
            'score': score
        })

    # Сохраняем результаты в CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Готово! Результаты сохранены в {OUTPUT_FILE}")


if __name__ == '__main__':
    main()

