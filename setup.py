from setuptools import setup

setup(
    name="keyverbum",
    version="0.0.1",
    description="Library of keywords extractors",
    url="http://github.com/kuparez/keyverbum",
    author="Nikita Churikov",
    author_email="nikita@chur.ru",
    license="MIT",
    packages=["keyverbum"],
    install_requires=[
        "pymorphy2",
        "nltk",
        "scipy",
        "sklearn",
        "gensim",
        "networkx",
        "numpy",
        "pytest",
        "yake",
        "requests",
        "flask",
    ],
    zip_safe=False,
)
