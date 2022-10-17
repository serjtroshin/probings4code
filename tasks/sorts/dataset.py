import json
from pathlib import Path

from data.github.preprocessing.src.code_tokenizer import (
    extract_functions_java, tokenize_java)
from src.datasets.dataset import Dataset

from .config import get_classification_task_parser


class LabeledDataset(Dataset):
    def __init__(self, args, type="sorts"):
        self.input_file = args.input_file
        self.data = load_data(self.input_file)
        self._type = type

    def __iter__(self):
        for sent, label in self.data:
            # sent = " ".join(tokenize_java(sent))
            yield sent, label

    def __len__(self):
        return len(self.data)


def load_data(path="codes_preprocessed"):
    for d in Path(path).glob("*"):
        name = d.name
        for code in d.glob("*"):
            with code.open("r") as f:
                yield f.read().strip(), name


# def load_data(path="dedup_code"):
#     for d in Path(path).glob("*"):
#         name = d.stem
#         with d.open("r") as code_file:
#             for line in code_file:
#                 code = json.loads(line)
#                 yield code, name

#                 # tokens = tokenize_java(code)
#                 # print(tokens)
#                 # for func in extract_functions_java(" ".join(tokens))[0]:
#                 #     # print(colored(func, "blue"))
#                 #     # print()
#                 #     yield func, name
#                 # for func in extract_functions_java(" ".join(tokens))[1]:
#                 #     # print(colored(func, "green"))
#                 #     # print()
#                 #     yield func, name
#     return


def get_dataset():
    """
    creates a dataset
    """
    PLBART_PATH = Path(__file__, "../../..").absolute().resolve()
    parser = get_classification_task_parser(PLBART_PATH)
    args = parser.parse_args([])
    return LabeledDataset(args)
