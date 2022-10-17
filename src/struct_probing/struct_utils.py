from typing import List

import torch


class NodeRepr:
    def __init__(self, vec):
        self.vec = vec

    def __str__(self):
        return f"vec:{str(self.vec)}"

    def dist(self, other):
        i = 0
        while i < len(self.vec) and i < len(other.vec) and self.vec[i] == other.vec[i]:
            i += 1
        return (len(self.vec) - i) + (len(other.vec) - i)


class ReprList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Sample:
    def __init__(self, data):
        self.data = data
        self.MAXTOKENS = 512

    def _crop(self, x):
        return x[: self.MAXTOKENS - 2]

    @property
    def code(self) -> str:
        # for this experiment we need precise alighnment with bpe
        return self.bpe.replace("▁", " ").strip()

    @property
    def bpe(self) -> str:
        # example: '▁public ▁static ▁void ▁setState ▁( ▁State ▁state ▁) ▁{ ▁currentState ▁= ▁state ▁; ▁}'
        return " ".join(self._crop(self.data["outputs"]["bpe"].strip().split()))

    @property
    def tokens(self) -> torch.Tensor:
        return self.data["outputs"]["tokens"][1:-1]

    @property
    def features(self) -> List[torch.Tensor]:
        return [layer[0, 1:-1, :] for layer in self.data["outputs"]["features"]]

    @property
    def representations(self) -> List[List]:
        return ReprList(
            NodeRepr(vec) for vec in self._crop(self.data["representations"])
        )

    def __str__(self):
        return f"code: {self.code}\nbpe: {self.bpe}\ntokens: {self.tokens}\nfeatures: List[torch.tensor]\nrep: {ReprList(str(v) for v in self.representations)}"

    def to_dict(self):
        return self.data

    def layers(self):
        return self.data["embeddings_info"]


def to_dataset(data):
    def _has_repr(x):
        try:
            x.representations
            return True
        except KeyError:
            return False

    dataset = [Sample(data) for data in data]
    dataset = list(filter(_has_repr, dataset))
    print("len(dataset))", len(dataset))
    return dataset
