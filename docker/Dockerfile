FROM pytorch/pytorch:latest

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN git clone https://github.com/OpenNMT/OpenNMT-py.git && cd OpenNMT-py && pip install -r requirements.txt && python setup.py install
RUN cd ../
RUN git clone https://github.com/kuparez/keyverbum.git && cd keyverbum && pip install .
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN git clone https://github.com/Mottl/ru_punkt.git && cp -r ru_punkt/nltk_data/tokenizers/punkt/PY3/russian.pickle ~/nltk_data/tokenizers/punkt/PY3/russian.pickle && cp -r ru_punkt/nltk_data/tokenizers/punkt/russian.pickle ~/nltk_data/tokenizers/punkt/russian.pickle

ENV SERVER_PORT 5000
EXPOSE "$SERVER_PORT"

# set default command
CMD ["python"]