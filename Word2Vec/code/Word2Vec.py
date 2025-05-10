"""
Обработка и обучение распределённых представлений слов (эмбеддинги)
с помощью Word2Vec (CBOW и Skip-Gram) на синтаксически предобработанном русском корпусе,
с последующим сравнением их качества через косинусное сходство и евклидово расстояние.
Шаги:
1) Проверка и загрузка UDPipe-модели для синтаксической предобработки.
2) Предобработка исходных текстов: токенизация, лемматизация и разметка POS с учётом именных групп.
3) Обучение двух моделей (CBOW и Skip-Gram) по каждому обработанному файлу:
4) Сравнение моделей по заранее заданному списку пар слов:
   - Косинусное сходство (cosine similarity)
   - Евклидово расстояние (Euclidean distance)
5) Сохранение:
   - предобработанных файлов в `output/processed/`
   - обученных моделей в `output/models/`
   - результаты сравнения в `output/results/model_comparison.csv`
"""
from pathlib import Path
import os
import wget
from ufal.udpipe import Model, Pipeline
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd

# Пути
BASE_DIR      = Path(__file__).resolve().parent.parent
PARSE_DIR     = BASE_DIR / 'parse'                    # Исходные тексты
OUTPUT_DIR    = BASE_DIR / 'output'
PROCESSED_DIR = OUTPUT_DIR / 'processed'              # Предобработанные тексты
MODEL_DIR     = OUTPUT_DIR / 'models'                 # Сохранённые модели
RESULTS_DIR   = OUTPUT_DIR / 'results'                # Итоговые таблицы
for d in (PROCESSED_DIR, MODEL_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# URL скачивания очищенной UDPipe-модели для русского
UDP_URL    = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
MODEL_FILE = MODEL_DIR / 'udpipe_syntagrus.model'

# Пары слов для сравнения моделей
WORD_PAIRS = [
    ('экзамен_NOUN', 'государственный_ADJ'),
    ('командировка_NOUN', 'капитан_NOUN'),
    ('командировка_NOUN', 'уехать_VERB'),
]

# Параметры обучения Word2Vec
VECTOR_SIZE = 500    # размерность векторов
WINDOW      = 5      # размер контекстного окна
MIN_COUNT   = 2      # минимальная частотность слов
TOP_N       = 10     # топ-N ключевых значений (если используется)

# Загрузка UDPipe
def download_udpipe():
    """
    Проверяет наличие модели на диске и загружает её при необходимости.
    """
    if not MODEL_FILE.exists():
        print('Downloading UDPipe model...')
        wget.download(UDP_URL, str(MODEL_FILE))
        print(f"\nModel downloaded to {MODEL_FILE}")
    else:
        print('UDPipe model already present.')

# Предобработка текста
def process(pipeline: Pipeline, text: str,
            keep_pos: bool=True, keep_punct: bool=False) -> str:
    """
    Синтаксическая разметка через UDPipe:
    - Лемматизация и POS-теги
    - Группировка имён собственных
    Возвращает строку из лемма_POS, разделённых пробелом.
    """
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = mem_num = None
    tagged = []

    # конвертация в CoNLL-U формат
    output = pipeline.process(text)
    lines = [l for l in output.split('\n') if not l.startswith('#') and l]

    for line in lines:
        cols = line.split('\t')
        if len(cols) != 10:
            continue
        _, token, lemma, pos, _, feats, _, _, _, _ = cols

        # Обработка имён собственных
        if pos in entities:
            morph = dict(el.split('=') for el in feats.split('|') if '=' in el)
            if not named:
                named = True
                mem_case = morph.get('Case')
                mem_num  = morph.get('Number')
                memory   = [lemma]
            elif morph.get('Case') == mem_case and morph.get('Number') == mem_num:
                memory.append(lemma)
            else:
                tagged.append('::'.join(memory) + '_PROPN')
                memory = [lemma]
            continue
        if named:
            tagged.append('::'.join(memory) + '_PROPN')
            named = False; memory = []

        # Пропуск чисел
        if pos == 'NUM' and token.isdigit():
            continue

        tagged.append(f"{lemma}_{pos}")

    # Фильтрация пунктуации
    if not keep_punct:
        tagged = [w for w in tagged if not w.endswith('_PUNCT')]
    # Оставить только леммы, без POS
    if not keep_pos:
        tagged = [w.rsplit('_', 1)[0] for w in tagged]

    return ' '.join(tagged)


def tag_ud(text: str, pipeline: Pipeline) -> str:
    """
    Разбивает текст на строки и обрабатывает каждую через process().
    """
    return '\n'.join(process(pipeline, line) for line in text.split('\n'))

# Предобработка всех файлов
def preprocess_all():
    # Загрузка модели и инициализация Pipeline
    model    = Model.load(str(MODEL_FILE))
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    for txt in PARSE_DIR.glob('*.txt'):
        text = txt.read_text(encoding='utf-8', errors='ignore')
        processed = tag_ud(text, pipeline)
        out_file = PROCESSED_DIR / f"processed_{txt.name}"
        out_file.write_text(processed, encoding='utf-8')
        print(f"Processed: {txt.name}")

# Обучение моделей CBOW и Skip-Gram
def train_models():
    for proc in PROCESSED_DIR.glob('processed_*.txt'):
        sentences = LineSentence(str(proc))
        # CBOW
        cbow = Word2Vec(sentences,
                        vector_size=VECTOR_SIZE,
                        window=WINDOW,
                        min_count=MIN_COUNT,
                        sg=0)
        # Skip-Gram
        sg = Word2Vec(sentences,
                      vector_size=VECTOR_SIZE,
                      window=WINDOW,
                      min_count=MIN_COUNT,
                      sg=1)

        cbow_path = MODEL_DIR / f"cbow_{proc.stem}.model"
        sg_path   = MODEL_DIR / f"skipgram_{proc.stem}.model"
        cbow.save(str(cbow_path)); print(f"Saved CBOW model: {cbow_path}")
        sg.save(str(sg_path));     print(f"Saved SG model: {sg_path}")

# Сравнение моделей
def compare_models():
    rows = []
    for proc in PROCESSED_DIR.glob('processed_*.txt'):
        stem      = proc.stem
        cbow      = Word2Vec.load(str(MODEL_DIR / f"cbow_{stem}.model"))
        sg        = Word2Vec.load(str(MODEL_DIR / f"skipgram_{stem}.model"))
        for w1, w2 in WORD_PAIRS:
            try:
                # Косинусное сходство
                cos_cb = cbow.wv.similarity(w1, w2)
                cos_sg = sg.wv.similarity(w1, w2)
                # Евклидово расстояние между эмбеддингами
                vec_cb1 = cbow.wv[w1]; vec_cb2 = cbow.wv[w2]
                vec_sg1 = sg.wv[w1];   vec_sg2 = sg.wv[w2]
                euc_cb = np.linalg.norm(vec_cb1 - vec_cb2)
                euc_sg = np.linalg.norm(vec_sg1 - vec_sg2)
                rows.append([
                    proc.name,
                    f"{w1} vs {w2}",
                    round(cos_cb, 4), round(cos_sg, 4),
                    round(euc_cb, 4), round(euc_sg, 4)
                ])
            except KeyError:
                print(f"Word not in vocab: {w1} or {w2}")
    df = pd.DataFrame(rows,
                      columns=[
                          'File',
                          'Word Pair',
                          'Cosine_CBOW', 'Cosine_SG',
                          'Euclid_CBOW', 'Euclid_SG'
                      ])
    out = RESULTS_DIR / 'model_comparison.csv'
    df.to_csv(out, index=False)
    print(f"Saved comparison results to {out}")

def main():
    download_udpipe()
    preprocess_all()
    train_models()
    compare_models()

if __name__ == '__main__':
    main()
