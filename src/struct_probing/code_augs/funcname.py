import logging
from tkinter import Label
from typing import Iterator, List, Tuple

import numpy as np

from .aug import (CodeAugmentation, CodeParser, LabeledToken, SentenceInfo,
                  Token, TokenTypes, Tree)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from src.struct_probing.utils.code_parser import TreeSitterCode, traverse_tree

from .aug import LabeledEdge, debuggable
from .constants import RequiredEmbeddings


class FuncNamePrediction(CodeAugmentation):
    def __init__(self, type="funcname"):
        super().__init__(type)
        self._type = type

    def required_dataset(self):
        return "funcname"

    def required_embeddings(self) -> RequiredEmbeddings:
        return RequiredEmbeddings.MEAN

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:

        if tokens[0].node.type == "@":
            return  # for simplisity filter out functions with decorators

        f_name = None
        replace_with = "[MASK]"
        labeled_tokens = []
        was_bracket = False
        for i, token in enumerate(tokens):
            if token.node.type == "{":
                was_bracket = True
            if (
                f_name is None
                and i < len(tokens) - 1
                and token.node.type == "identifier"
                and tokens[i + 1].node.type == "("
            ):
                if was_bracket:
                    continue  # it is not the beggining of a function declaration
                # function name
                f_name = token.value
                labeled_tokens.append(LabeledToken(replace_with, 1))
            else:
                # if token.node.type == "identifier" and f_name is not None and token.value == f_name:
                #     # to not leak f_name occasionally
                #     labeled_tokens.append(LabeledToken(replace_with, 0))
                # else:
                labeled_tokens.append(LabeledToken(token.value, 0))
        if f_name is None:
            return
        yield labeled_tokens, [], SentenceInfo(f_name)
