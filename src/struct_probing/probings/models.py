from enum import Enum


class ProbingModelType(Enum):
    LINEAR = "linear"
    MLP = "mlp"
    LOWERBOUND = "lowerbound"
    LOWERBOUND_NGRAM = "lowerbound_ngram"
    BOW = "bow"

    # magic methods for argparse compatibility

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)
