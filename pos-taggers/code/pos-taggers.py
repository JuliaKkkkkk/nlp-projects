"""
Сравнение POS-тэггеров pymorphy3 и spaCy для русского текста
"""
from pathlib import Path
import csv
from pymorphy3 import MorphAnalyzer
from pymorphy3.tokenizers import simple_word_tokenize
import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'parse'
OUTPUT_DIR = BASE_DIR / 'output'

# Инициализация анализаторов
morph = MorphAnalyzer()
# Загружаем модель spaCy ru_core_news_sm, загружаем её при отсутствии
try:
    nlp = spacy.load('ru_core_news_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('ru_core_news_sm')
    nlp = spacy.load('ru_core_news_sm')


def load_text(file_path: Path) -> str:
    """
    Загружает текст из файла.
    Пробует utf-8-sig, затем cp1251.
    """
    try:
        return file_path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        return file_path.read_text(encoding='cp1251')


def tag_pymorphy(text: str) -> list[tuple[str, str]]:
    """
    Токенизация и POS-теггинг через pymorphy3.
    Возвращает список кортежей (token, POS_tag).
    """
    tokens = simple_word_tokenize(text)
    tags = []
    for tok in tokens:
        parsed = morph.parse(tok)[0]
        pos = parsed.tag.POS or 'X'
        tags.append((parsed.word, pos))
    return tags


def tag_spacy(text: str) -> list[tuple[str, str]]:
    """
    POS-теггинг через spaCy.
    Возвращает список (token, pos_), где pos_ — универсальная POS-тег.
    """
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def save_to_csv(results: dict[str, list[tuple[str, str]]], prefix: str):
    """
    Сохраняет результаты теггинга в CSV-файлы.
    Для каждого метода создаёт файл output/{prefix}_{method}.csv
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for method, tags in results.items():
        path = OUTPUT_DIR / f"{prefix}_{method}.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['token', 'pos'])
            for token, pos in tags:
                writer.writerow([token, pos])


def main():
    """
    Основная функция: обходит все тексты в папке parse/,
    применяет два метода теггинга и сохраняет результаты.
    """
    for file_path in sorted(INPUT_DIR.glob('*.txt')):
        text = load_text(file_path)
        pym = tag_pymorphy(text)
        sp = tag_spacy(text)
        save_to_csv({'pymorphy3': pym, 'spacy': sp}, file_path.stem)
        print(f"Processed: {file_path.name}")


if __name__ == '__main__':
    main()

"""
Сравнение результатов разметки

1. Имена собственные
Pymorphy3: Относит "Брюсов" к существительному, даже если это имя собственное.
Spacy: Корректно маркирует имена собственные как PROPN (например, "Брюсов" и "В").

2. Предлоги и союзы
Pymorphy3: Обозначает предлоги как PREP.
Spacy: Обозначает предлоги как ADP.

3. Ошибки

Spacy:
26. кормила - VERB, должно быть существительное (pymorphy3 и spaCy)
29. закатный  - VERB, должно быть прилагательное
32-34. странно-кос -  разбито на части: "странно" помечено как прилагательное, "-", как существительное, "кос" — как имя собственное, хотя это одно прилагательное
40. синели - NOUN, должен быть глагол (pymorphy3 и spaCy)
45. кроя - NOUN, должен быть глагол(pymorphy3 и spaCy)
64. строго-четки - разбито на части: - "строго" и "-" помечены как наречие, "четки" как существительное, хотя это одно прилагательное
67./167. миг - имя собственное, должно быть существительное
71. жал - VERB, должно быть существительное (pymorphy3 и spaCy)
84. весло - VERB, должно быть существительное
103. закат - имя собственное, должно быть существительное
110. Алея - имя собственное, должен быть герундий
127. уста - имя собственное, должно быть существительное
179. ничтожная - VERB, должно быть прилагательное
184. 12 - ADV, должно быть числительное
186. 1914 - ADV, должно быть числительное

pymorphy3
в.я. - предлоги и пунктуация, должны бфть имена собственные
24. кормила - VERB, должно быть существительное (pymorphy3 и spaCy)
34. синели - NOUN, должен быть глагол (pymorphy3 и spaCy)
36. вечеровой - NOUN, должно быть прилагательное
38. кроя - NOUN, должен быть глагол (pymorphy3 и spaCy)
52. строго-четки - NOUN, должно быть прилагательное
71. жал - VERB, должно быть существительное (pymorphy3 и spaCy)
99. правда - PRCL, должно быть существительное

SpaCy лучше справляется с выделением имен собственных (но иногда отмечает существительные как имена собственные), в инициалах не выделяет точки как отдельные токены, но хуже справляется с разметкой
Pymorphy3 лучше справляется с разметкой, не выделяет слова с дефисом как отдельные, но плохо справляется с выделением имен собственных
"""