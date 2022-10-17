from pathlib import Path

import pandas as pd
from data.github.preprocessing.src.code_tokenizer import tokenize_java
from src.datasets.dataset import Dataset

from .config import get_classification_task_parser


class LabeledDataset(Dataset):
    def __init__(self, args, type="readability"):
        self.input_file = args.input_file
        self.data = load_data(self.input_file)
        self._type = type

    def __iter__(self):
        for sent, label in self.data:
            sent = " ".join(tokenize_java(sent))
            yield sent, label

    def __len__(self):
        return len(self.data)


def load_data(path="Dataset"):
    # mean score
    df = pd.read_csv(f"{str(path)}/scores.csv")
    data = []
    for name in Path(path).glob("**/*.jsnp"):
        id_ = name.name[:-5]
        data.append((name.open("r").read(), df[f"Snippet{id_}"].mean()))
    return data


def get_dataset():
    """
    creates a dataset
    """
    PLBART_PATH = Path(__file__, "../../..").absolute().resolve()
    parser = get_classification_task_parser(PLBART_PATH)
    args = parser.parse_args([])
    return LabeledDataset(args)
