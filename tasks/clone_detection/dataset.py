import argparse
from pathlib import Path

from src.datasets.dataset import Dataset

from models.utils import SMPEncoder

from .config import get_classification_task_parser

"""
Usage:
from clone_detection.dataset import get_dataset
dataset = get_dataset(plbart_path)
"""


class PairDataset(Dataset):
    """__iter__ returns a pair of two sentences, and a target"""

    def __init__(self, args, type="code_clone"):
        assert type in PairDataset.available()
        self.args = args
        self._type = type

    @classmethod
    def available(cls):
        return ["code_clone"]

    def __iter__(self):
        return getattr(PairDataset, self.type)(self)

    def code_clone(self):
        for sent1, sent2, lab in iter_test_data(self.args):
            yield {"sent1": sent1, "sent2": sent2}, int(lab)

    def __len__(self):
        return len_test_data(self.args)


def get_dataset(plbart_path, args):
    """
    creates a dataset
    """
    parser = get_classification_task_parser(plbart_path)
    args = parser.parse_args(args)
    return PairDataset(args)


def iter_test_data(args):
    args, encoder = args, SMPEncoder(args.sentencepiece)
    with open(args.input_file) as inpf, open(args.label_file) as labelf:
        acc = 0
        n = 0
        for i, (inp, lab) in enumerate(zip(inpf, labelf)):
            sent1, sent2 = inp.split("</s>")
            sent1, sent2 = encoder.decode(sent1.split()), encoder.decode(sent2.split())
            yield sent1, sent2, lab


def len_test_data(args):
    with open(args.label_file) as inpf:
        return len(inpf.readlines())
