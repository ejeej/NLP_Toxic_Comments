import functools
import re
import pymorphy2
import pandas as pd
from razdel import sentenize
from tqdm.notebook import tqdm
from transliterate import translit
from nltk.corpus import stopwords

mystopwords = stopwords.words('russian') 
mystopwords.remove('хорошо')
add_stopwords = pd.read_fwf('stopwords-ru.txt', header=None)
add_stopwords = add_stopwords[0].tolist()
mystopwords += add_stopwords
mystopwords = list(set(mystopwords))

regex = re.compile("[А-Яа-яA-z]+")

def words_only(text, regex=regex):
    try:
        return regex.findall(re.sub('w', 'в', re.sub('q', 'кв', translit(re.sub('ё', 'e', text.lower()), 'ru'))))
    except:
        return []

m = pymorphy2.MorphAnalyzer()

@functools.lru_cache(maxsize=128)
def lemmatize_word(token, pymorphy=m):
    return re.sub('ё', 'e', pymorphy.parse(token)[0].normal_form)

def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]

def remove_stopwords(lemmas, stopwords=mystopwords):
    return [w for w in lemmas if not w in stopwords and len(w) > 2]

def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    
    return ' '.join(remove_stopwords(lemmas))

def sentence_list(text):
    return [_.text for _ in list(sentenize(text))]

def clean_sentence_text(text):
    sentences = sentence_list(text)
    tokens = [words_only(sent) for sent in sentences]
    lemmas = [lemmatize_text(tok) for tok in tokens]
    
    return [remove_stopwords(lem) for lem in lemmas]