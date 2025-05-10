#!/usr/bin/env python3
"""
Снятие синтаксической неоднозначности в русском предложении двумя подходами:
1) Морфологически расширенная FCFG с ViterbiParser
2) Статистическая PCFG с InsideChartParser и ViterbiParser
"""
from pathlib import Path
import codecs
from nltk import word_tokenize
from nltk.parse import ViterbiParser, load_parser
from nltk.grammar import PCFG
from nltk.parse import pchart
import pymorphy3 as pm

# Пути
CODE_DIR = Path(__file__).resolve().parent
PARSE_DIR = CODE_DIR.parent / 'parse'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
RULES_FILE = PARSE_DIR / 'rules.txt'
FCFG_FILE  = PARSE_DIR / 'test.fcfg'

# Морфологический анализатор
m = pm.MorphAnalyzer()

# Сбрасываем и записываем базовые правила
with codecs.open(FCFG_FILE, mode='w', encoding='utf-8') as f_out:
    with codecs.open(RULES_FILE, mode='r', encoding='utf-8') as f_rules:
        f_out.writelines(f_rules.readlines())


def pm3fcfg(phrase):
    """
    Дописывает в test.fcfg морфологические правила для каждого слова.
    """
    with codecs.open(FCFG_FILE, mode='a', encoding='utf-8') as f:
        for x in phrase:
            parses = m.parse(x)
            if not parses:
                print(f"Error: No parse found for {x}")
                continue
            for y in parses:
                if y.tag.POS in ("NOUN", "ADJF", "PRTF"):
                    strk = f"{y.tag.POS}[C={y.tag.case}, G={y.tag.gender}, NUM={y.tag.number}, PER=3, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                elif y.tag.POS in ("ADJS", "PRTS"):
                    strk = f"{y.tag.POS}[G={y.tag.gender}, NUM={y.tag.number}, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                elif y.tag.POS == "NUMR":
                    strk = f"{y.tag.POS}[C={y.tag.case}, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                elif y.tag.POS in ("ADVB", "GRND", "COMP", "PRED", "PRCL", "INTJ"):
                    strk = f"{y.tag.POS}[NF=u'{y.normal_form}'] -> '{y.word}'\n"
                elif y.tag.POS in ("PREP", "CONJ"):
                    strk = f"{y.tag.POS}[NF=u'{y.normal_form}'] -> '{y.word}'\n"
                    f.write(strk)
                    break
                elif y.tag.POS == "NPRO" and y.normal_form not in ("это", "нечего"):
                    if y.tag.person and y.tag.person[0] == '3' and y.tag.number == 'sing':
                        strk = f"{y.tag.POS}[C={y.tag.case}, G={y.tag.gender}, NUM={y.tag.number}, PER={y.tag.person[0]}, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                    else:
                        strk = f"{y.tag.POS}[C={y.tag.case}, NUM={y.tag.number}, PER={y.tag.person[0]}, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                elif y.tag.POS in ("VERB", "INFN"):
                    if y.tag.tense == 'past':
                        strk = f"{y.tag.POS}[TR={y.tag.transitivity}, TENSE={y.tag.tense}, G={y.tag.gender}, NUM={y.tag.number}, PER=0, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                    elif y.tag.POS == 'INFN':
                        strk = f"{y.tag.POS}[TR={y.tag.transitivity}, TENSE=0, G=0, NUM=0, PER=0, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                    else:
                        strk = f"{y.tag.POS}[TR={y.tag.transitivity}, TENSE={y.tag.tense}, G=0, NUM={y.tag.number}, PER={y.tag.person[0]}, NF=u'{y.normal_form}'] -> '{y.word}'\n"
                else:
                    continue
                f.write(strk)


def test_ambiguity(text, max_trees=2):
    """
    Тест синтаксической неоднозначности по FCFG.
    Сохраняет результаты в output/fcfg_parses.txt.
    """
    words = word_tokenize(text.lower())
    fcfg_out = OUTPUT_DIR / 'fcfg_parses.txt'
    with fcfg_out.open('w', encoding='utf-8') as outf:
        outf.write(f"Токены: {words}\n")
        pm3fcfg(words)
        cp = load_parser(str(FCFG_FILE), trace=0)
        trees = list(cp.parse(words))
        if len(trees) > 1:
            outf.write(f"Обнаружены неоднозначности: {len(trees)} разборов\n")
            for i, tree in enumerate(trees[:max_trees], 1):
                outf.write(f"Разбор {i} из {max_trees} (максимум):\n{tree}\n")
        elif len(trees) == 1:
            outf.write("Неоднозначности не обнаружены. Один разбор:\n")
            outf.write(f"{trees[0]}\n")
        else:
            outf.write("Неоднозначности не обнаружены\n")


def pcfg_parsing(text):
    """
    PCFG-разбор с InsideChartParser и ViterbiParser.
    Сохраняет результаты в output/pcfg_parses.txt.
    """
    grammar_with_penalty = PCFG.fromstring("""
        S    -> NP VP                     [1.0]    
        VP   -> V                         [0.01]
        VP   -> V NP                      [0.01]
        VP   -> V PP                      [0.01]
        VP   -> V NP PP                   [0.01]
        VP   -> V PP PP                   [0.96]  
        NP   -> Name                      [0.49]
        NP   -> N                         [0.49]
        NP   -> Det N                     [0.01]
        NP   -> NP PP                     [0.01]
        PP   -> P NP                      [1.0] 
        V    -> 'пошёл'                   [1.0]
        N    -> 'стадион'                 [0.5]
        N    -> 'собакой'                 [0.5]
        Name -> 'Джон'                    [1.0]
        P    -> 'на'                      [0.5]
        P    -> 'с'                       [0.5]
    """
    )
    tokens = text.split()
    out_file = OUTPUT_DIR / 'pcfg_parses.txt'
    with out_file.open('w', encoding='utf-8') as outf:
        outf.write("Деревья с применением штрафов:\n")
        inside = pchart.InsideChartParser(grammar_with_penalty)
        vt = ViterbiParser(grammar_with_penalty)
        for t in inside.parse(tokens):
            outf.write(f"Дерево разбора с InsideChartParser. Вероятность: {t.prob():.4f}:\n{t}\n")
        for t in vt.parse(tokens):
            outf.write(f"Дерево разбора с ViterbiParser. Вероятность: {t.prob():.4f}:\n{t}\n")


def main():
    text = "Джон пошёл на стадион с собакой"
    test_ambiguity(text)
    pcfg_parsing(text)

if __name__ == '__main__':
    main()
