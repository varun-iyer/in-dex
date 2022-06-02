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

def sent2vec(sentences):
    from sent2vec.vectorizer import Vectorizer
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    vectors = vectorizer.vectors
    return vectors

def cosine_graph(sentences, skip_adjacent=0):
    """
    :param skip_adjacent: skips a pairing if their difference is less than
    """
    vectors = sent2vec(sentences)
    graph = {}
    for (n1, s1), (n2, s2), in it.combinations(enumerate(vectors), 2):
        if abs(n1 - n2) < skip_adjacent:
            continue
        graph[frozenset((n1, n2))] = 1 - cosine(s1, s2)
    return graph

def tf_idf(sentences):
    ti = TfidfVectorizer()
    X = ti.fit_transform(sentences)
    return X

def helpful_links(sentences, method: ("sent2vec", "tfidf")="sent2vec"):
    graph = cosine_graph(sentences, skip_adjacent=10)
    ordered = sorted(graph.items(), key=lambda x: x[1])[::-1]
    n = int(len(sentences)/10)
    return ordered[:n]


# see https://github.com/plangrid/pdf-annotate
if __name__ == "__main__":
    # k = helpful_links(katz)
    tf_idf(katz)
