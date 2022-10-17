import logging
import random
import warnings
from collections import defaultdict
from email.policy import default
from tkinter import Label
from typing import Any, Iterator, List, Optional, Tuple

import torch
from src.models.model import ModelOutput

from .aug import (CodeAugmentation, CodeParser, LabeledToken, SentenceInfo,
                  Token, TokenTypes, Tree)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from src.struct_probing.utils.code_parser import TreeSitterCode, traverse_tree

from .aug import LabeledEdge, debuggable


class TokenLabel:
    MISUSED = "MISUSED"
    ORIG = "ORIG"


class SentLabel:
    ERROR = "ERROR"
    ORIG = "ORIG"


class VarMisuseAugmentation(CodeAugmentation):
    def __init__(self, type="varmisuse"):
        super().__init__(type)
        self._type = type

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:

        labeled_tokens = []
        unique_identifiers_2_id = defaultdict(list)
        all_place_holders: List[Tuple[int, str]] = list()
        all_unique_values = set()
        for it, token in enumerate(tokens):
            if token.node.type == TokenTypes.Identifier:
                unique_identifiers_2_id[token.value].append(it)
                all_place_holders.append((it, token.value))
                all_unique_values.add(token.value)
        if len(unique_identifiers_2_id) == 0:
            return

        # random.shuffle(unique_identifiers)  # shuffle identifier ids

        MAX_SELECTIONS = 3
        random_pairs_of_vars = []  # pairs [token_id, value to replace]

        random.shuffle(all_place_holders)
        for token_id, value in all_place_holders:
            for other_value in unique_identifiers_2_id.keys():
                if other_value != value:
                    random_pairs_of_vars.append((token_id, other_value))

        random.shuffle(random_pairs_of_vars)
        random_pairs_of_vars = random_pairs_of_vars[:MAX_SELECTIONS]

        for (token_id, value_to_replace) in random_pairs_of_vars:
            # create varmisuse sequence
            labeled_tokens = []
            for it, token in enumerate(tokens):
                if it == token_id:
                    assert token.node.type == TokenTypes.Identifier
                    assert token.value != value_to_replace
                    labeled_tokens.append(
                        LabeledToken(value_to_replace, TokenLabel.MISUSED)
                    )  # copy var_2
                    # mark var_1 placeholder as wrong
                else:
                    labeled_tokens.append(LabeledToken(token.value, 0))
            yield labeled_tokens, [], SentenceInfo(SentLabel.ERROR)

            # add true var
            labeled_tokens = []
            for it, token in enumerate(tokens):
                if it == token_id:
                    labeled_tokens.append(LabeledToken(token.value, TokenLabel.ORIG))
                    # mark var_1 placeholder as wrong
                else:
                    labeled_tokens.append(LabeledToken(token.value, 0))
            yield labeled_tokens, [], SentenceInfo(SentLabel.ORIG)

    def embeddings_hook(
        self,
        model_output: ModelOutput,
        labeled_bpe: Optional[List[Any]],
        labeled_edges: List[LabeledEdge],
    ):
        # remove 0 label embeddings
        # print(model_output.bpe,
        #       model_output.tokens.shape,
        #       model_output.hiddens[0].shape)
        # print(len(labeled_bpe), labeled_bpe)
        ids_remain: torch.LongTensor = torch.tensor(
            [
                i
                for i, x in enumerate(labeled_bpe)
                if x in [TokenLabel.ORIG, TokenLabel.MISUSED]
            ]
        )
        if torch.numel(ids_remain) == 0:
            return None, None, None
        # print(ids_remain)
        model_output = model_output.filter_by_token_ids(ids_remain)
        labeled_bpe = [labeled_bpe[i] for i in ids_remain]
        # print("-------")
        # print(model_output.bpe,
        #       model_output.tokens.shape,
        #       model_output.hiddens[0].shape)
        # input()
        return model_output, labeled_bpe, labeled_edges
