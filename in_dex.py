import itertools as it
from scipy.spatial.distance import cosine
from nltk.tokenize import sent_tokenize, word_tokenize
from stop_words import get_stop_words
import re

stop_words = get_stop_words('en')
legal_stop_words = ["u.s.", "v."]
stop_words += legal_stop_words

word = re.compile("[a-zA-Z]+")

test_sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]

def tokdoc(raw_text):
    raw_text = raw_text.replace("\n", " ").lower()
    tok = sent_tokenize(raw_text)
    tok = [[w for w in word_tokenize(sentence) if (w not in stop_words and re.match(word, w))] for sentence in tok]
    return tok

katz = open("katz.txt").read().replace("U. S.", "U.S.")

def nwise(iterable, n=2):
    """
    An implementation of itertools.pairwise for triples and other sets

    for 'ABCDEFG', nwise('ABCDEFG', 3) -> 'ABC', 'BCD', 'CDE'...
    """
    return zip(*[iterable[i::n] for i in range(n)])

def ngram_tfidf(tokenized_sentences, sentence_ngram=1, word_ngram=1):
    df = {}
    tf = {}
    for sentences in nwise(tokenized_sentences, sentence_ngram):
        document = sum(sentences, [])
        words = [" ".join(w) for w in nwise(document, word_ngram)]
        wordcount = {}
        for w in words:
            wordcount[w] = wordcount.get(w, 0) + 1
        for word, occurences in wordcount.items():
            tf[word] = tf.get(word, 0) + occurences
            df[word] = df.get(word, 0) + 1
    n_documents = len(sentences) + 1 - sentence_ngram
    idf = {k: n_documents / v for k, v in df.items() if v > 1}
    average_tf = {k: v / df[k] for k, v in tf.items() if v > 1}
    keys = set.intersection(set(idf.keys()), set(average_tf.keys()))
    return {key: idf[key] * average_tf[key] for key in keys}
 
def fuzzy_search(term, text):
    return list(re.finditer(r".{,3}\s(\w+.{,3}\s){,5}".join(term), text.lower()))

def key_phrases(document):
    tok = tokdoc(document)
    sorted_phrases = sum([list(ngram_tfidf(tok, 4, n).keys()) for n in range(4, 1, -1)], [])
    n_sentence = len(tok)
    sorted_phrases = sorted_phrases[:n_sentence//5]
    return sorted_phrases

def in_dex(document):
    sorted_phrases = key_phrases(document)
    link_sets = [fuzzy_search(sp.split(" "), document) for sp in sorted_phrases]
    return link_sets

# see https://github.com/plangrid/pdf-annotate
if __name__ == "__main__":
    k = in_dex(katz)
     
