import logging
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from src.struct_probing.probings.base import ProbingDataset, ProbingModel, ProbingTask
# from sklearn.neural_network import MLPRegressor
from src.struct_probing.probings.mlp_utils import TorchMLPRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.struct_probing.utils.code_parser import CodeParser
from src.struct_probing.utils.sample import Sample
from src.struct_probing.utils.tree_representation import get_node_representations, is_identifier

# log = logging.Logger("sentence_regression", level=logging.INFO)


class SentenceRegressionLinearModel(ProbingModel):
    def __init__(self):
        pass

    def train_linear_model(self, X_train, y_train):
        model = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100))
        model.fit(np.array(X_train), np.array(y_train))
        logging.info(str(model))
        return model

    def predict_linear_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def train_upper_bound_model(self, X_train, y_train):
        model = TorchMLPRegressor()
        model.fit(np.array(X_train), np.array(y_train))
        return model

    def predict_upper_bound_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def eval_prediction(self, y_true, y_pred) -> dict:
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "1 - r2": 1 - r2_score(y_true, y_pred),
        }

    def train_lower_bound(
        self, samples: List[Sample], y_train: List[Any]
    ) -> Callable[[Sample], Any]:
        def bpe_len(sample: Sample):
            print(sample.bpe)
            input()
            return len(sample.bpe.split(" "))

        bpes_lens = np.array(list(map(bpe_len, samples)))[:, None]
        model = LassoCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100))
        model.fit(bpes_lens, y_train)

        def predict(_sample: Sample):
            bpe_lens = np.array([bpe_len(_sample)])[:, None]
            assert bpe_lens.shape == (1, 1), bpe_lens.shape
            return model.predict(bpe_lens)[0]

        return predict


class Dataset(ProbingDataset):
    def __init__(self, X: Dict[int, Any], y: List[Any], samples: List[Sample]):
        self._X = X
        self._y = y
        self._samples = samples

    @property
    def X_by_layer(self) -> Dict[int, Any]:
        return self._X

    @property
    def y(self) -> Any:
        return self._y

    @property
    def samples(self) -> List[Sample]:
        return self._samples


class RegressionProbingTask(ProbingTask):
    def __init__(
        self,
        name: str,
        get_target: Callable[[Any], Union[float, int]],
        description="",
        embedding_type="mean",
    ):
        self.name = name
        self.description = description.strip()
        self._get_target = get_target
        self.embedding_type = embedding_type

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_target(self, x: str) -> Union[float, int]:
        return self._get_target(x)

    def get_probing_model(self) -> SentenceRegressionLinearModel:
        return SentenceRegressionLinearModel()

    def get_embedding_type(self) -> str:
        return self.embedding_type

    def _make_dataset(
        self,
        train_data: List[Sample],
        test_data: List[Sample],
        layers: List[int],
        **kwargs
    ) -> Tuple[Dataset, Dataset]:
        logging.info("make dataset")

        def do(data: List[Sample], layers):
            X_by_layer = defaultdict(list)
            y_list = []
            for data_sample in data:
                if len(data_sample.data["outputs"]["features"][0].shape) == 1:
                    features = data_sample.features(handle="ident")
                else:
                    features = data_sample.features(handle="mean")
                for layer in layers:
                    assert len(features) > layer, (len(features), layers)
                    X_by_layer[layer].append(features[layer].numpy())
                y_list.append(self.get_target(data_sample))

            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
            y_numpy = np.array(y_list)
            return X_by_layer_numpy, y_numpy

        X_train_by_layer, y_train = do(train_data, layers)
        print(y_train)

        X_test_by_layer, y_test = do(test_data, layers)
        assert len(train_data) == len(y_train)
        return Dataset(X_train_by_layer, y_train, train_data), Dataset(
            X_test_by_layer, y_test, test_data
        )


"""
Sentence Regression Probing Tasks:
"""


def get_depth(sample: Sample) -> int:
    if sample.true_code is not None:
        code = sample.true_code
    else:
        code = sample.code
    # max depth of AST tree
    code_tree = CodeParser("java")(code)
    nodes = get_node_representations(code_tree.tree)
    nvs = list(filter(lambda p: is_identifier(p.node), nodes))
    if len(nvs) == 0:
        return 0
    max_depth = max(len(p.representation) for p in nvs)
    return max_depth


# def get_log_ntokens(sample: Sample) -> float:
#     code = sample.code
#     # log of number of tokens in a code
#     return math.log(1 + len(code.split(" ")))


PROBINGS: List[ProbingTask] = [
    RegressionProbingTask(
        "get_depth",
        get_depth,
        description="""
            Maximal depth of the AST tree
        """,
        embedding_type="dummy",
    ),
    # RegressionProbingTask(
    #     "get_log_ntokens",
    #     get_log_ntokens,
    #     description="""
    #         Log(1 + x), where x is the number of bpe tokens (TODO)
    #     """,
    #     embedding_type="dummy"
    # ),
]


if __name__ == "__main__":
    sample = Sample.default()
    for task in PROBINGS:
        print("Task:", task.get_name())
        print("Description:", task.get_description())
        print("Sample:", sample.bpe)
        print("Targets:", task.get_target(sample))
