from http import HTTPStatus

import pymorphy2
from flask import Flask, request, jsonify

from keyverbum import keywords

app = Flask(__name__)


@app.route("/v0/keywords", methods=["POST"])
def get_keywords():
    try:
        assert request.json["text"], "Invalid text"
        assert request.json["algorithm"], "Invalid algorithm"
        assert request.json["n_keywords"], "Invalid keywords number"

        inp = request.json["text"]
        algorithm = request.json["algorithm"]
        n_keywords = int(request.json["n_keywords"])

        text = keywords.preprocessing_pipeline.transform(inp)
        if algorithm == "tfidf":
            extractor = keywords.TfIdf(n_keywords=n_keywords)
            text = [" ".join(text)]
        elif algorithm == "topicrank":
            morph = pymorphy2.MorphAnalyzer()
            stemmer = keywords.PymorphyStemmer(morph)
            extractor = keywords.TopicalPagerank(
                n_keywords=n_keywords, morph=morph, stemmer=stemmer
            )
        elif algorithm == "yake":
            extractor = keywords.YAKE(n_keywords=n_keywords)
            text = " ".join(text)
        elif algorithm == "textrank":
            extractor = keywords.Textrank(n_keywords=n_keywords)
            text = " ".join(text)
        else:
            raise KeyError("Invalid algorithm")

        result = extractor.fit(text).predict(text)
        return jsonify({"keywords": result})

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
