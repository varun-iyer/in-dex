import itertools as it
from scipy.spatial.distance import cosine
from nltk.tokenize import sent_tokenize, word_tokenize
from stop_words import get_stop_words
import re

stop_words = get_stop_words('en')

word = re.compile("\w+")

test_sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]

def tokdoc(raw_text):
    raw_text = raw_text.replace("\n", " ")
    tok = sent_tokenize(raw_text)
    tok = [[w for w in word_tokenize(sentence) if (w not in stop_words and re.match(word, w))] for sentence in tok]
    return tok

katz = tokdoc(open("katz.txt").read())

def nwise(iterable, n=2):
    """
    An implementation of itertools.pairwise for triples and other sets

    for 'ABCDEFG', nwise('ABCDEFG', 3) -> 'ABC', 'BCD', 'CDE'...
    """
    return zip(*[tokenized_sentences[i::n] for i in range(n)])

def ngram_tfidf(tokenized_sentences, sentence_ngram=1, word_ngram=1):
    df = {}
    tf = {}
    for sentences in nwise(tokenized_sentences, sentence_ngram):
        document = sum(sentences)
        words = [" ".join(w) for w in nwise(document, word_ngram)]
        wordcount = {}
        for w in words:
            wordcount[w] = wordcount.get(w, 0) + 1
        for word, occurences in wordcount.items():
            tf[word] = tf.get(word, 0) + occurences
            df[word] = tf.get(word, 0) + 1
    n_documents = sentences + 1 - sentence_ngram
    idf = {k: n_documents / v for k, v in df.items() if v > 1}
    average_tf = {k: v / df[k] for k, v in tf.items() if v / df[k] > 1}
    keys = set().intersection(idf.keys(), average_tf.keys())
    return {key: idf[key] * average_tf[key] for key in keys}
 
def helpful_links(sentences, method: ("sent2vec", "tfidf")="sent2vec"):
    graph = cosine_graph(sentences, skip_adjacent=10)
    ordered = sorted(graph.items(), key=lambda x: x[1])[::-1]
    n = int(len(sentences)/10)
    return ordered[:n]


# see https://github.com/plangrid/pdf-annotate
if __name__ == "__main__":
    # k = helpful_links(katz)
    ngram_tfidf(katz)
