import logging
import re
from collections import defaultdict
from itertools import product
from typing import List, Tuple, Union, Optional, Iterable

import networkx as nx
import nltk
import pymorphy2
from gensim.summarization.keywords import keywords
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BasicPreprocessor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        text = re.sub(r"<[^>]*>", "", X)
        text = re.sub(r"[\W]+", " ", text.lower())
        return text


class Stemmer(TransformerMixin):
    def fit(self, X: str, y=None):
        return self

    def transform(self, X, y):
        return NotImplementedError()


class NoneStemmer(Stemmer):
    def fit(self, X, y=None):
        return None

    def transform(self, X: str, y=None):
        return X.lower()


class PymorphyStemmer(Stemmer):
    def __init__(self, morph):
        self.morph: pymorphy2.MorphAnalyzer = morph

    def fit(self, X, y=None):
        return self

    def transform(self, X: str, y=None):
        return self.morph.normal_forms(X)[0]


class StopwordsFilter(TransformerMixin):
    def __init__(self, lang: str):
        try:
            self.stopwords = set(stopwords.words(lang))
        except LookupError as e:
            logger.warning(f"Could not load nltk stopwords for lang {lang}. {e}")
            self.stopwords = {}

    def fit(self, X, y=None):
        return self

    def transform(
        self, X: Union[Iterable[str], Iterable[Iterable[str]]], y=None
    ) -> Optional[Union[str, List[str], List[List[str]]]]:
        res: Optional[Union[str, List[str], List[List[str]]]] = []
        if not isinstance(X, str):
            for text in X:
                out = self.transform(text)
                if out is not None:
                    res.append(out)
        else:
            if X not in self.stopwords:
                res = X
            else:
                res = None
        return res


class KeywordsExtractor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        raise NotImplementedError()


class Textrank(KeywordsExtractor):
    def __init__(
        self, ratio=0.2, words=None, split=True, scores=False, pos_filter=("NN", "JJ"), deacc=True
    ):
        self.deacc = deacc
        self.pos_filter = pos_filter
        self.scores = scores
        self.split = split
        self.words = words
        self.ratio = ratio

    def predict(self, X: str, y=None) -> Union[List[Tuple[str, float]], List[str], str]:
        return keywords(
            X,
            ratio=self.ratio,
            words=self.words,
            split=self.split,
            scores=self.scores,
            pos_filter=self.pos_filter,
            lemmatize=False,
            deacc=self.deacc,
        )


class TopicalPagerank(KeywordsExtractor):
    def __init__(
        self,
        morph: pymorphy2.MorphAnalyzer,
        stemmer: Stemmer,
        preprocessor: BasicPreprocessor,
        stopwords: StopwordsFilter,
        lang: str,
    ):
        self.lang = lang
        self.stopwords = stopwords
        self.preprocessor = preprocessor
        self.tag_set = {"ADJF", "ADJS", "NOUN", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"}

        # mapping of keyphrases to its positions in text
        self.phrases = defaultdict(list)
        self.unstem_map = {}
        self.morph = morph
        self.stemmer = stemmer

        self.text = []
        self.topics = []

    def fit(self, X: str, y=None):
        """Fit keywords extractor for single text"""
        for sent in sent_tokenize(self.preprocessor.transform(X), language=self.lang):
            for word in word_tokenize(sent, language=self.lang):
                self.text.append(word)
        return self

    def _extract_phrases(self):
        phrases = [[]]
        positions = []
        counter = 0
        for token in self.text:
            p = self.morph.parse(token)[0]
            if str(p.tag) == "LATN":
                _, pos = nltk.pos_tag([token])[0]
            else:
                pos = p.tag.POS

            if pos in self.tag_set:
                stemmed_word = self.stemmer.transform(token)
                if stemmed_word and len(stemmed_word) > 1:
                    phrases[-1].append(stemmed_word)
                    self.unstem_map[stemmed_word] = (counter, token)
                if len(phrases[-1]) == 1:
                    positions.append(counter)
            else:
                if phrases[-1]:
                    phrases.append([])
            counter += 1
        for n, phrase in enumerate(phrases):
            if phrase:
                self.phrases[" ".join(sorted(phrase))] = [
                    i for i, j in enumerate(phrases) if j == phrase
                ]
        logger.debug("Found {} keyphrases".format(len(self.phrases)))

    def calc_distance(self, topic_a, topic_b):
        """
        Calculate distance between 2 topics
        :param topic_a: list if phrases in a topic A
        :param topic_b: list if phrases in a topic B
        :return: int
        """
        result = 0
        for phrase_a in topic_a:
            for phrase_b in topic_b:
                if phrase_a != phrase_b:
                    phrase_a_positions = self.phrases[phrase_a]
                    phrase_b_positions = self.phrases[phrase_b]
                    for a, b in product(phrase_a_positions, phrase_b_positions):
                        result += 1 / abs(a - b)
        return result

    def _identify_topics(self, strategy="average", max_d=0.75):
        """
        Group keyphrases to topics using Hierarchical Agglomerative Clustering (HAC) algorithm
        :param strategy: linkage strategy supported by scipy.cluster.hierarchy.linkage
        :param max_d: max distance for cluster identification using distance criterion in scipy.cluster.hierarchy.fcluster
        :return: None
        """
        # use term freq to convert phrases to vectors for clustering
        count = CountVectorizer()
        bag = count.fit_transform(list(self.phrases.keys()))

        # apply HAC
        Z = linkage(bag.toarray(), strategy)
        c, coph_dists = cophenet(Z, pdist(bag.toarray()))
        if c < 0.8:
            logger.warning("Cophenetic distances {} < 0.8".format(c))

        # identify clusters
        clusters = fcluster(Z, max_d, criterion="distance")
        cluster_data = defaultdict(list)
        for n, cluster in enumerate(clusters):
            inv = count.inverse_transform(bag.toarray()[n])
            cluster_data[cluster].append(
                " ".join(sorted([str(i) for i in count.inverse_transform(bag.toarray()[n])[0]]))
            )
        logger.debug("Found {} keyphrase clusters (topics)".format(len(cluster_data)))
        topic_clusters = [frozenset(i) for i in cluster_data.values()]
        # apply pagerank to find most prominent topics
        # Sergey Brin and Lawrence Page. 1998.
        # The Anatomy of a Large - Scale Hypertextual Web Search Engine.
        # Computer Networks and ISDN Systems 30(1): 107–117
        topic_graph = nx.Graph()
        topic_graph.add_weighted_edges_from(
            [
                (v, u, self.calc_distance(v, u))
                for v in topic_clusters
                for u in topic_clusters
                if u != v
            ]
        )
        pr = nx.pagerank(topic_graph, weight="weight")

        # sort topic by rank
        self.topics = sorted([(b, list(a)) for a, b in pr.items()], reverse=True)

    def predict(
        self, n: int, y=None, cluster_strategy="average", max_d=1.25, extract_strategy="first"
    ):
        """
        Get topN topic based n ranks and select
        :param n: topN
        :param strategy: How to select keyphrase from topic:
                         -first - use the one which appears first
                         -center - use the center of the cluster WIP
                         -frequent - most frequent WIP
        :return: list of most ranked keyphrases
        """
        result = []
        self._extract_phrases()
        self._identify_topics(strategy=cluster_strategy, max_d=max_d)
        if extract_strategy != "first":
            logger.warning("Using 'first' extract_strategy to extract keyphrases")
        for rank, topic in self.topics[:n]:
            if topic:
                first_kp = topic[0]  # sorted(topic, key=lambda x: self.phrases[x][0])[0]
                unstem_kp_sort = sorted([self.unstem_map[i] for i in first_kp.split(" ")])
                unstem_kp = " ".join([i[1] for i in unstem_kp_sort])
                result.append(unstem_kp)
        return result


class TfIdf(KeywordsExtractor):
    def __init__(
        self,
        tokens_filter: StopwordsFilter,
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        n_keywords=10,
    ):

        self.n_keywords = n_keywords
        self.tokens_filter = tokens_filter
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range, max_df=max_df, min_df=min_df, smooth_idf=True, use_idf=True
        )
        self.feature_names = None

    def __sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        # use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        # results = zip(feature_vals,score_vals)
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    def fit(self, X, y=None):
        self.tfidf.fit(X, y)
        self.feature_names = self.tfidf.get_feature_names()
        return self

    def predict(self, X, y=None):
        tfidf_vec = self.tfidf.transform(X)
        sorted_items = self.__sort_coo(tfidf_vec.tocoo())

        keywords = self._extract_topn_from_vector(self.feature_names, sorted_items, self.n_keywords)

        return keywords
