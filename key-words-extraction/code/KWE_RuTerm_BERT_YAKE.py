"""
Извлечение ключевых выражений из полного текста и аннотации
с помощью RuTerm, YAKE и KeyBERT, результаты сохраняются в Excel.
"""
from pathlib import Path
import pandas as pd
from rutermextract import TermExtractor
from keybert import KeyBERT
import yake

# Пути
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
INPUT_DIR = BASE_DIR / 'parse'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Файлы корпуса
TEXT_FILE = INPUT_DIR / 'Vissio_IMS_2022.txt'
ABSTRACT_FILE = INPUT_DIR / 'Vissio_IMS_2022_Abstract_rus.txt'

class KeywordExtractor:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.bert_model = KeyBERT()

    def RuTerm(self, text: str):
        extractor = TermExtractor()
        terms = extractor(text)
        return [(t.normalized, t.count) for t in terms[:self.top_n]]

    def YAKE(self, text: str):
        kw_ext = yake.KeywordExtractor(
            lan="ru", n=3, top=self.top_n,
            dedupLim=0.9, dedupFunc='seqm', windowsSize=1
        )
        return kw_ext.extract_keywords(text)

    def BERT(self, text: str, ngram_range=(1,1)):
        return self.bert_model.extract_keywords(
            text, keyphrase_ngram_range=ngram_range, top_n=self.top_n
        )

    def save_to_excel(self, data, filename):
        df = pd.DataFrame(data, columns=["Term", "Score"])
        path = OUTPUT_DIR / filename
        df.to_excel(path, index=False)
        print(f"Saved: {path}")

    def save_combined_excel(self, data_abstract, data_text, filename):
        df_abs = pd.DataFrame(data_abstract, columns=["Term", "Score"]).assign(Source="Abstract")
        df_txt = pd.DataFrame(data_text,     columns=["Term", "Score"]).assign(Source="Text")
        max_len = max(len(df_abs), len(df_txt))
        df_abs = df_abs.reindex(range(max_len))
        df_txt = df_txt.reindex(range(max_len))
        df_comb = pd.concat([df_abs, df_txt], axis=1)
        path = OUTPUT_DIR / filename
        with pd.ExcelWriter(path) as writer:
            df_comb.to_excel(writer, sheet_name="Keywords", index=False)

def main():
    extractor = KeywordExtractor(top_n=10)
    # Чтение текстов
    text = TEXT_FILE.read_text(encoding='utf-8')
    abstract = ABSTRACT_FILE.read_text(encoding='utf-8')

    # Экстракция
    rut_abs    = extractor.RuTerm(abstract)
    rut_txt    = extractor.RuTerm(text)
    yake_abs   = extractor.YAKE(abstract)
    yake_txt   = extractor.YAKE(text)
    bert_uni_abs = extractor.BERT(abstract, ngram_range=(1,1))
    bert_uni_txt = extractor.BERT(text,     ngram_range=(1,1))
    bert_bi_abs  = extractor.BERT(abstract, ngram_range=(1,2))
    bert_bi_txt  = extractor.BERT(text,     ngram_range=(1,2))
    bert_tri_abs = extractor.BERT(abstract, ngram_range=(1,3))
    bert_tri_txt = extractor.BERT(text,     ngram_range=(1,3))

    # Сохранение
    extractor.save_combined_excel(rut_abs,    rut_txt,    "RuTerm.xlsx")
    extractor.save_combined_excel(yake_abs,   yake_txt,   "YAKE.xlsx")
    extractor.save_combined_excel(bert_uni_abs, bert_uni_txt, "KeyBERT_unigram.xlsx")
    extractor.save_combined_excel(bert_bi_abs,  bert_bi_txt,  "KeyBERT_bigram.xlsx")
    extractor.save_combined_excel(bert_tri_abs, bert_tri_txt, "KeyBERT_trigram.xlsx")

if __name__ == '__main__':
    main()
