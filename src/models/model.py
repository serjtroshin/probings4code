from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class ModelOutput:
    def __init__(
        self, bpe: List[str], tokens: torch.Tensor, hiddens: List[torch.Tensor]
    ):

        assert tokens.ndim == 1
        assert tokens.shape[0] == len(bpe), (tokens.shape[0], len(bpe), bpe)
        assert all(h.ndim == 3 for h in hiddens)  # batch, len, emb_dim
        assert all(h.shape[1] == tokens.shape[0] for h in hiddens)

        self.bpe = bpe
        self.tokens = tokens
        self.hiddens = hiddens

    def filter_by_token_ids(self, ids_remain: torch.LongTensor) -> ModelOutput:
        assert len(ids_remain.shape) == 1
        return ModelOutput(
            bpe=[self.bpe[i] for i in ids_remain],
            tokens=self.tokens[ids_remain.numpy()],
            hiddens=[hidden[:, ids_remain, :] for hidden in self.hiddens],
        )

    def dump(self):
        return {
            "bpe": " ".join(self.bpe),
            "tokens": self.tokens,
            "features": self.hiddens,
        }


class Model(ABC):
    def __init__(self, type):
        self._type = type

    @property
    def type(self):
        return self._type

    @staticmethod
    @abstractmethod
    def get_model():
        pass

    @abstractmethod
    def bpe(self, code: str) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        pass

    @abstractmethod
    def __call__(self, code: str) -> ModelOutput:
        pass
