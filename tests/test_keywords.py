import pymorphy2

from keyverbum.keywords import (
    Textrank,
    TopicalPagerank,
    PymorphyStemmer,
    BasicPreprocessor,
    StopwordsFilter,
    TfIdf,
)

TEXT_1 = (
    "Challenges in natural language processing frequently involve"
    "speech recognition, natural language understanding, natural language"
    "generation (frequently from formal, machine-readable logical forms),"
    "connecting language and machine perception, dialog systems, or some"
    "combination thereof."
)
KEYWORDS_1 = ["natural language", "machine"]

TEXT_2 = (
    "Inverse problems for a mathematical model of ion exchange in a "
    "compressible ion exchanger. A mathematical model of ion exchange "
    "is considered, allowing for ion exchanger compression in the process "
    "of ion exchange. Two inverse problems are investigated for this model, "
    "unique solvability is proved, and numerical solution methods are proposed. "
    "The efficiency of the proposed methods is demon strated by a numerical experiment"
)
KEYWORDS_2 = ["exchang ion", "mathemat model"]


def test_textrank_keywords():

    key_extractor = Textrank()
    assert set(key_extractor.predict(TEXT_1)) == set(KEYWORDS_1)


def test_example():
    morph = pymorphy2.MorphAnalyzer()
    stemmer = PymorphyStemmer(morph)
    preprocessor = BasicPreprocessor()
    sw_filter = StopwordsFilter("russian")
    tr = TopicalPagerank(
        morph=morph, stemmer=stemmer, preprocessor=preprocessor, stopwords=sw_filter, lang="russian"
    )
    tr.fit(TEXT_2)
    res = tr.predict(n=2)
    assert res == ["ion exchange", "mathematical model"]


def test_tfidf():
    sw_filter = StopwordsFilter("russian")
    tr = TfIdf(tokens_filter=sw_filter, n_keywords=2, ngram_range=(1, 2))
    tr.fit([TEXT_2])
    res = tr.predict([TEXT_2])
    assert [k[0] for k in sorted(res.items(), key=lambda x: x[1], reverse=True)] == ["ion", "of"]
