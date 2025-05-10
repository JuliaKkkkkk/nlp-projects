"""
Сравнение токенизаторов русского текста.
Скрипт обрабатывает входной файл, прогоняет несколько методов токенизации,
сохраняет результаты в текстовые файлы и формирует две сводные таблицы:
- counts.csv — количество токенов для каждого метода
- tokens.csv — полная таблица с токенами разных методов
"""
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('stanza').setLevel(logging.WARNING)

from itertools import zip_longest
from pathlib import Path
import pandas as pd

import nltk
from razdel import tokenize as razdel_tokenize
from segtok.tokenizer import word_tokenizer as segtok_tokenize
from pymorphy3 import tokenizers as pym_tokenizers
from spacy.lang.ru import Russian
import stanza
import ufal.udpipe
from mosestokenizer import MosesTokenizer

# Пути к файлам и директориям
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / 'parse' / 'Чехов.txt'
MODEL_FILE = BASE_DIR / 'parse' / 'russian-syntagrus-ud-2.0-170801.udpipe'
OUTPUT_DIR = BASE_DIR / 'output'

def tokenize_nltk(text: str) -> list[str]:
    """Токенизация через NLTK"""
    from nltk import word_tokenize
    return word_tokenize(text)


def tokenize_razdel(text: str) -> list[str]:
    """Токенизация через Razdel"""
    return [tok.text for tok in razdel_tokenize(text)]


def tokenize_segtok(text: str) -> list[str]:
    """Токенизация через Segtok"""
    return list(segtok_tokenize(text))


def tokenize_pymorphy(text: str) -> list[str]:
    """Токенизация через Pymorphy3"""
    return pym_tokenizers.simple_word_tokenize(text)


def tokenize_spacy(text: str) -> list[str]:
    """Токенизация через spaCy"""
    nlp = Russian()
    doc = nlp(text)
    return [token.text for token in doc]


def tokenize_stanza(text: str) -> list[str]:
    """Токенизация через Stanza (UDPipe)"""
    nlp = stanza.Pipeline(lang='ru', processors='tokenize', use_gpu=False, verbose=False)
    doc = nlp(text)
    return [w.text for sent in doc.sentences for w in sent.words]


def tokenize_moses(text: str) -> list[str]:
    """Токенизация через MosesTokenizer"""
    with MosesTokenizer('ru') as mtok:
        tokens = []
        for line in text.splitlines():
            tokens.extend(mtok(line))
        return list(tokens)


def tokenize_udpipe(text: str) -> list[str]:
    """Токенизация через модель UDPipe"""
    model = ufal.udpipe.Model.load(str(MODEL_FILE))
    pipeline = ufal.udpipe.Pipeline(
        model, 'tokenize',
        ufal.udpipe.Pipeline.DEFAULT,
        ufal.udpipe.Pipeline.DEFAULT,
        ufal.udpipe.Pipeline.DEFAULT
    )
    processed = pipeline.process(text)
    tokens = []
    for line in processed.split('\n'):
        parts = line.split('\t')
        if len(parts) > 1:
            tokens.append(parts[1])
    return tokens


def diff_func(arr1: list[str], arr2: list[str], name1: str, name2: str):
    """Сравнение уникальных токенов двух списков"""
    set1, set2 = set(arr1), set(arr2)
    diff1 = set1 - set2
    diff2 = set2 - set1
    print(f"\nСравниваем {name1} vs {name2}: ")
    print(f"В {name1}, но не в {name2}: {diff1}")
    print(f"В {name2}, но не в {name1}: {diff2}")


def main():
    # Проверка наличия входных файлов
    if not INPUT_FILE.exists():
        print(f"Файл с текстом не найден: {INPUT_FILE}")
        return
    if not MODEL_FILE.exists():
        print(f"UDPipe модель не найдена: {MODEL_FILE}")
        return

    # Чтение входного текста
    text = INPUT_FILE.read_text(encoding='utf-8-sig')

    # Создание директории вывода
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Токенизация всеми методами
    arrays = {
        'nltk': tokenize_nltk(text),
        'razdel': tokenize_razdel(text),
        'segtok': tokenize_segtok(text),
        'pymorphy': tokenize_pymorphy(text),
        'spacy': tokenize_spacy(text),
        'stanza': tokenize_stanza(text),
        'moses': tokenize_moses(text),
        'ufal': tokenize_udpipe(text),
    }

    # Сохранение результатов в файлы
    for name, tokens in arrays.items():
        (OUTPUT_DIR / f"{name}.txt").write_text("\n".join(tokens), encoding='utf-8')

    # Сравнения
    diff_func(arrays['nltk'], arrays['razdel'], 'nltk', 'razdel')
    diff_func(arrays['segtok'], arrays['pymorphy'], 'segtok', 'pymorphy')
    diff_func(arrays['stanza'], arrays['spacy'], 'stanza', 'spacy')
    diff_func(arrays['moses'], arrays['ufal'], 'moses', 'ufal')

    # Формирование сводных таблиц
    counts = {name: len(tokens) for name, tokens in arrays.items()}
    df_counts = pd.DataFrame({'tokenizer': list(counts.keys()), 'count': list(counts.values())})
    df_counts.to_csv(OUTPUT_DIR / 'counts.csv', index=False)

    rows = list(zip_longest(*arrays.values(), fillvalue=''))
    df_tokens = pd.DataFrame(rows, columns=arrays.keys())
    df_tokens.to_csv(OUTPUT_DIR / 'tokens.csv', index=False)


if __name__ == '__main__':
    main()
