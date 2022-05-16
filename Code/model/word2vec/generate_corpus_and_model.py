import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

PLAY_FILES_DIR = 'Raw_Data/plays/'
CORPUS_DIR = 'Data/corpus.txt'

def build_corpus():
    play_files = os.listdir(PLAY_FILES_DIR)
    corpus_file = open(CORPUS_DIR, 'w')
    for file in play_files:
        if not file.endswith(".txt"):
            continue
        with open(PLAY_FILES_DIR + file) as f:
            text = f.read()
        if not file.endswith("processed.txt"):
            text = text[text.index('ACT 1'):]
        for i in sent_tokenize(text):
            i = re.sub('[^a-zA-Z0-9 \n]', '', i.lower())
            words = word_tokenize(i)
            corpus_file.write(f"{' '.join(words)}\n")

def load_and_pretrain_model():
    corpus = open(CORPUS_DIR)
    for count, line in enumerate(corpus):
        pass
    model = Word2Vec(corpus_file=CORPUS_DIR, vector_size=200, window=5, min_count=1)
    model.save('w2v-plays.model')
    model.wv.save('w2v-plays.wv')

build_corpus()
load_and_pretrain_model()