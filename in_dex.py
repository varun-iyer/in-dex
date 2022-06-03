import itertools as it
from scipy.spatial.distance import cosine
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
import re
from bs4 import BeautifulSoup

stop_words = get_stop_words("en")
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
    ps = PorterStemmer()
    tok = [
        [
            w
            for w in word_tokenize(sentence)
            if (w not in stop_words and re.match(word, w) and len(w) > 3)
        ]
        for sentence in tok
    ]
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
    n_documents = len(tokenized_sentences) + 1 - sentence_ngram
    idf = {k: n_documents / v for k, v in df.items() if v > 1}
    average_tf = {k: v / df[k] for k, v in tf.items() if v > 1}
    keys = set.intersection(set(idf.keys()), set(average_tf.keys()))
    return {key: idf[key] * average_tf[key] for key in keys}


def fuzzy_search(term, text):
    return list(re.finditer(r".{,3}\s(\w+.{,3}\s){,5}".join(term), text, re.IGNORECASE))


def key_phrases(document):
    tok = tokdoc(document)
    sorted_phrases = sum(
        [list(ngram_tfidf(tok, 4, n).keys()) for n in range(2, 1, -1)], []
    )
    n_sentence = len(tok)
    sorted_phrases = sorted_phrases[: n_sentence // 5]
    return sorted_phrases


def in_dex(document, sorted_phrases=None):
    if not sorted_phrases:
        sorted_phrases = key_phrases(document)
    link_sets = [fuzzy_search(sp.split(" "), document) for sp in sorted_phrases]
    return link_sets


def html_wrap(html, span: (0, -1), tag: ("<a>", "</a>")):
    """
    Wrap a given span of html with a tag, returning the number of characters
    added.
    """
    out_html = (
        html[: span[0]]
        + tag[0]
        + html[span[0] : span[1]]
        + tag[1]
        + html[span[1] :]
    )
    return out_html


def in_dex_html(html_file):
    bs = BeautifulSoup(open(html_file))
    out_html = str(bs.html)
    indices = in_dex(out_html, key_phrases(bs.text))
    annotations = []
    for nset, match_set in enumerate(indices):
        for nmatch, match in enumerate(match_set):
            annotations.append((nset, nmatch, match))
    annotations = sorted(annotations, key=lambda x: match.span()[0])

    for nset, nmatch, match in annotations:
            start = match.span()[0]
            while rematch := re.search(re.escape(str(match.group())), out_html[start:], re.IGNORECASE):
                if out_html[start + rematch.span()[0] - 1] != ">" and \
                        out_html[start + rematch.span()[1]] != ">":
                    break
                start += len(rematch.group())

            if not rematch:
                print("No rematch")
                continue
            link_up = ""
            link_down = ""
            if nmatch - 1 >= 0:
                link_up = f'<a href="#{nset}_{max(nmatch-1, 0)}"><sup>&uarr;</sup></a>'
            if nmatch + 1 <= len(indices[nset]):
                link_down = f'<a href="#{nset}_{min(nmatch+1, len(indices[nset])-1)}" class="noline"><sup>&darr;</sup></a>'
            out_html = html_wrap(
                out_html,
                (start + rematch.span()[0], start + rematch.span()[1]),
                (f'<a id="{nset}_{nmatch}">', f'</a>' + link_up + link_down)
            )
    outfile = open(f"{html_file}.idx", "w")
    from css import css
    outfile.write(f"<!DOCTYPE html> <html><head><style> {css} </style></head>")
    outfile.write(out_html)
    return out_html

if __name__ == "__main__":
    import sys
    in_dex_html(sys.argv[1])
