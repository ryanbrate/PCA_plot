# global imports
import pathlib

from gensim.models import Word2Vec


class Vectors:
    """A class for returning an object with access to gensim W2V vectors."""

    def __init__(self, model_fp: pathlib.Path):
        self.model = Word2Vec.load(str(model_fp))

    def __getitem__(self, embedding_label: str):
        return self.model.wv[embedding_label]

    def has_label(self, label):
        if label in self.model.wv:
            return True
        else:
            return False

