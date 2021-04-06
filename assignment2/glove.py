import numpy as np

class Glove():
    """
    Load GloVE embeddings from file.
    @attribute wv: the embedding matrix.
    @attribute size: embeddings size.
    """
    def __init__(self, path_to_glove_file):
        self.wv = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.wv[word] = coefs
        self.size = len(next(iter(self.wv.values())))
