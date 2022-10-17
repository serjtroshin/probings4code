from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
from src.utils import load_data

from src.struct_probing.utils.code_parser import CodeParser
from src.struct_probing.utils.match_tokens import match_bpe, match_nodes
from src.struct_probing.utils.tree_representation import (NodeRepr, NodeRepresentation,
                                       get_node_representations)


class Sample:
    def __init__(self, data: dict, validate=True):
        """
        Args:
            data (dict): dict from load_data
            features_handle (str[none, mean]): postprocess of features (e.g. aggregation)
        """
        self.data = data
        self._nodes: Optional[List[NodeRepresentation]] = None
        self._representations: Optional[List[NodeRepr]] = None
        self.MAXTOKENS = 512

        meta = self.data["outputs"]
        if validate:
            if meta["bpe"] is not None and meta["tokens"] is not None:
                if len(meta["tokens"]) != len(meta["bpe"].strip().split()):
                    print(meta["tokens"])
                    print(meta["bpe"].strip().split())
                assert len(meta["tokens"]) == len(meta["bpe"].strip().split()), (
                    len(meta["tokens"]),
                    len(meta["bpe"].strip().split()),
                )
                # assert all(
                #     meta["features"][i].shape[1] == len(meta["tokens"])
                #     for i in range(len(meta["features"]))
                # )
                # assert len(self.bpe.strip().split()) == len(self.tokens), (self.bpe.strip().split(), len(self.tokens), len(self.bpe.strip().split()))
                # assert len(self.tokens) == len(self.features("none")[0]), (
                #     len(self.tokens), len(self.features("none")[0])
                # )
                # assert len(self.nodes) == len(meta["tokens"])
            else:
                warnings.warn("bpe or tokens is None")

    def _crop(self, x: List) -> Union[List, torch.Tensor]:
        """crop by max lendth (PLBART works with 512 tokens)"""
        return x[: self.MAXTOKENS - 2]

    @property
    def code(self) -> str:
        """code string from bpe tokens

        Returns:
            str: joined subtokens sequence with special token removed
        """
        # for this experiment we need precise alighnment with bpe
        return self.bpe.replace(" ", "").replace("▁", " ").strip()

    @property
    def bpe(self) -> str:
        """cropped bpe sequence (crop by MAXTOKENS=512)
        Returns:
            str: joined subtokens
        """
        # example: '▁public ▁static ▁void ▁setState ▁( ▁State ▁state ▁) ▁{ ▁currentState ▁= ▁state ▁; ▁}'
        if self.data["outputs"]["bpe"] is not None:
            return " ".join(self._crop(self.data["outputs"]["bpe"].strip().split()))
        return ""

    @property
    def true_bpe(self) -> str:
        # ablation
        if "true_bpe" in self.data and self.data["true_bpe"] is not None:
            return " ".join(self._crop(self.data["true_bpe"].strip().split()))
        else:
            return None

    @property
    def true_code(self) -> str:
        """code string from bpe tokens

        Returns:
            str: joined subtokens sequence with special token removed
        """
        # for this experiment we need precise alighnment with bpe
        if (
            "code_joined_correct" in self.data
            and self.data["code_joined_correct"] is not None
        ):
            return " ".join(
                self._crop(self.data["code_joined_correct"].strip().split())
            )
        else:
            return None

    @property
    def tokens(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor
        """
        if self.data["outputs"]["tokens"] is not None:
            return self.data["outputs"]["tokens"]
        return torch.tensor([])

    @property
    def default_target(self):
        """
        Returns:
            Any: a label from initial dataset
        """
        return self.data["target"]

    def features(self, handle: str) -> List[torch.Tensor]:
        """
        Args:
            handle (str, optional): none or mean. Defaults to "none".
            retrieve sentence embedding or token embeddings

        Returns:
            List[torch.tensor]: a tensor with embeddings for each layer
        """
        # handle(array[L, DIM]) -> array
        handle_f: Callable = {
            "none": lambda x: x[0],
            "mean": lambda x: x[0].mean(0),
            "ident": lambda x: x,
        }[handle]
        return [handle_f(layer) for layer in self.data["outputs"]["features"]]

    def _lazy_parse(self):
        if self._representations is not None and self._nodes is not None:
            return
        # print(self.code.strip())
        # print(self.bpe.strip().split())
        # input()
        if self.true_code is not None:
            sent = self.true_code.strip()
        else:
            sent = self.code.strip()
        code = CodeParser("java")(sent)
        nodes = list(get_node_representations(code.tree))

        tokens_bpe = self.bpe.strip().split()
        tokens = self.code.strip().split()

        bpe2token = match_bpe(tokens_bpe, tokens)
        token2node = match_nodes(sent, tokens, nodes)

        nodes_matched = []
        node_repr_matched = []
        for i, _bpe_token in enumerate(tokens_bpe):
            node = nodes[token2node[bpe2token[i]]]
            nodes_matched.append(node)
            node_repr_matched.append(NodeRepr(node.representation))
        self._representations = node_repr_matched
        self._nodes = nodes_matched

    @property
    def representations(self) -> Union[List[List[int]], List[NodeRepr]]:
        self._lazy_parse()  # -> self._nodes, self._representations
        assert self._representations is not None
        representations: List[NodeRepr] = self._representations
        return representations

    @property
    def nodes(self) -> List[NodeRepresentation]:
        self._lazy_parse()  # -> self._nodes, self._representations
        assert self._nodes is not None
        nodes: List[NodeRepresentation] = self._nodes
        return nodes

    @property
    def bpe_aug_labels(self) -> List[Any]:
        # per token labels
        labels = self.data["aug_labels"]
        assert len(labels) == len(self.bpe.split()), (
            len(labels),
            len(self.bpe.split()),
            labels,
            self.bpe.split(),
        )
        return labels

    @property
    def sentence_label(self) -> Any:
        return self.data["aug_info"]["label"]

    @property
    def edge_labels(self) -> Any:
        return self.data["labeled_edges"]

    @property
    def sentence_id(self) -> int:
        return self.data["sent_id"]

    def __str__(self):
        return f"code: {self.code}\nbpe: {self.bpe}\ntokens: {self.tokens}\nfeatures: List[torch.tensor]\nrep: {(str(v) for v in self.representations)}"

    @property
    def layers(self):
        return self.data["embeddings_info"]

    def to_dict(self):
        return self.data


def has_repr(x):
    try:
        x.representations
        return True
    except KeyError:
        return False


if __name__ == "__main__":
    print(Sample.default())
