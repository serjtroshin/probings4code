from pathlib import Path

from src.datasets.dataset import Dataset

from .config import get_classification_task_parser

"""
Usage:
from clone_detection.dataset import get_dataset
dataset = get_dataset(plbart_path)
"""


class SingleDataset(Dataset):
    """__iter__ returns a pair of two sentences, and a target"""

    def __init__(self, args, type="funcname"):
        self.args = args
        self._type = type

    def __iter__(self):
        for sent1 in iter_test_data(self.args):
            yield sent1, None  # None label

    def __len__(self):
        return len_test_data(self.args)


class AugDataset(Dataset):
    """__iter__ returns a meta"""

    def __init__(self, dataset, type, limit=1000000):
        self.dataset = dataset
        self._type = type
        self.limit = limit

    def __iter__(self):
        for it, sent1 in enumerate(self.dataset):
            yield sent1

            if it == self.limit:
                break

    def __len__(self):
        return self.dataset.__len__()


def get_dataset():
    """
    creates a dataset
    """
    PLBART_PATH = Path(__file__, "../../..").absolute().resolve()
    parser = get_classification_task_parser(PLBART_PATH)
    args = parser.parse_args([])
    return SingleDataset(args)


def iter_test_data(args):
    with open(args.input_file) as inpf:
        for sent1 in inpf:
            yield sent1.strip()


def len_test_data(args):
    with open(args.input_file) as inpf:
        return len(inpf.readlines())
