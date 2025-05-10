"""
Сравнение четырех методов стемминга для немецких слов:
Porter, Snowball, Lancaster, Regexp.
"""
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, RegexpStemmer
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer(language='german')
regexp = RegexpStemmer('e$|n$|er$|s$|st$|t$|en$|et$|est$|in$|erin$|', min=3)

words = ['Fische','Blumen','Kinder', 'Parks', 'höre', 'badest','arbeitet', 'reisen', 'gefahren', 'Lehrerin', 'Schüler']
print("{0:20}{1:20}{2:20}{3:30}{4:40}".format("Word","Porter Stemmer","Snowball Stemmer","Lancaster Stemmer",'Regexp Stemmer'))
for word in words:
    print("{0:20}{1:20}{2:20}{3:30}{4:40}".format(word,porter.stem(word),snowball.stem(word),lancaster.stem(word),regexp.stem(word)))

"""
Вывод: 
SnowballStemmer даёт наиболее осмысленные основы: 
Blumen→blum, reisen→reis, gefahren→gefahr. Это оптимальный выбор для немецкого стемминга.

PorterStemmer и LancasterStemmer изначально разработаны для английского. 
Они иногда вообще не обрабатывают многие немецкие формы (badest→badest), и не убирают Umlaut.

RegexpStemmer - элементарный, просто обрезает указанные суффиксы; 
при этом не приводит к нижнему регистру и может быть слишком грязным.
"""


# e$|n$|er$|s$|- множественное число
# der Fisch -  die Fische
# die Blume - die Blumen
# das Kind - die Kinder
# der Park – die Parks

# спряжение глаголов в настоящем времени
# e$|st$|t$|en$|et$|est$|
# höre,  badest, arbeitet, reisen

#словообразование жр и мр
# in$|erin$|er$|
# Lehrerin, Schüler
