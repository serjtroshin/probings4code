import logging
from tkinter import Label
from typing import Iterator, List, Tuple

import numpy as np

from .aug import CodeAugmentation, LabeledToken, SentenceInfo, Token, Tree

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from .aug import LabeledEdge, debuggable
from .constants import RequiredEmbeddings


class ReadabilityPrediction(CodeAugmentation):
    def __init__(self, type="readability"):
        super().__init__(type)
        self._type = type

    def required_dataset(self) -> str:
        return "readability"

    def required_embeddings(self) -> RequiredEmbeddings:
        return RequiredEmbeddings.MEAN

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:

        return super().__call__(tokens, ast)
