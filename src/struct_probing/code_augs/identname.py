import logging
import warnings
from tkinter import Label
from typing import Iterator, List, Tuple

import numpy as np

from .aug import (CodeAugmentation, CodeParser, LabeledToken, SentenceInfo,
                  Token, TokenTypes, Tree)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from src.struct_probing.utils.code_parser import TreeSitterCode, traverse_tree

from .aug import LabeledEdge, debuggable

TOP_IDENTIFIERS = set(
    map(
        lambda x: x[0],
        [
            ("result", 692),
            ("i", 370),
            ("value", 294),
            ("sb", 207),
            ("name", 195),
            ("c", 188),
            ("response", 187),
            ("key", 186),
            ("s", 185),
            ("builder", 179),
            ("t", 174),
            ("list", 168),
            ("b", 158),
            ("data", 149),
            ("p", 145),
            ("request", 145),
            ("id", 141),
            ("r", 140),
            ("index", 140),
            ("type", 132),
            ("f", 128),
            ("size", 122),
            ("count", 118),
            ("m", 117),
            ("file", 117),
            ("out", 116),
            ("res", 114),
            ("buf", 114),
            ("message", 113),
            ("in", 112),
            ("e", 111),
            ("v", 111),
            ("a", 108),
            ("x", 104),
            ("entry", 102),
            ("reader", 102),
            ("n", 102),
            ("other", 97),
            ("line", 96),
            ("status", 94),
            ("map", 94),
            ("url", 93),
            ("config", 92),
            ("params", 91),
            ("msg", 91),
            ("l", 86),
            ("expected", 85),
            ("val", 83),
            ("conf", 83),
            ("factory", 83),
        ],
    )
)


class IdentNamePrediction(CodeAugmentation):
    def __init__(self, type="identname"):
        super().__init__(type)
        self._type = type

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:

        labeled_tokens = []
        unique_identifiers = []
        for token in tokens:
            if (
                token.node.type == TokenTypes.Identifier
                and token.value in TOP_IDENTIFIERS
            ):
                unique_identifiers.append(token.value)
        if len(unique_identifiers) == 0:
            return

        random_identifier_idx = np.random.randint(0, len(unique_identifiers))
        random_identifier = unique_identifiers[random_identifier_idx]

        for token in tokens:
            if (
                token.node.type == TokenTypes.Identifier
                and token.value == random_identifier
            ):
                labeled_tokens.append(LabeledToken("var", random_identifier))
            else:
                labeled_tokens.append(LabeledToken(token.value, 0))

        yield labeled_tokens, [], SentenceInfo(None)
