import logging
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from src.struct_probing.probings.base import ProbingDataset, ProbingModel, ProbingTask
# from sklearn.neural_network import MLPRegressor
from src.struct_probing.probings.mlp_utils import TorchMLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.struct_probing.utils.sample import Sample

log = logging.Logger("token_regression", level=logging.INFO)


class TokenRegressionLinearModel(ProbingModel):
    def __init__(self):
        pass

    def train_linear_model(self, X_train, y_train):
        model = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100))
        model.fit(np.array(X_train), np.array(y_train))
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
        self, samples: List[str], y_train: List[Any]
    ) -> Callable[[str], Any]:
        assert len(samples) == len(y_train)
        bpe2labels: Dict[str, list] = defaultdict(list)
        for bpe, label in zip(samples, y_train):
            bpe2labels[bpe].append(label)
        global_mean = np.median(y_train)
        bpe2mean = {bpe: np.median(values) for bpe, values in bpe2labels.items()}

        def model(bpe: str) -> Any:
            if bpe in bpe2mean:
                return bpe2mean[bpe]
            else:
                return global_mean  # handle OOV bpe token

        return model


class Dataset(ProbingDataset):
    def __init__(self, X: Dict[int, Any], y: List[Any], samples: List[str]):
        self._X = X
        self._y = y
        self._samples = samples  # list of bpe tokens

    @property
    def X_by_layer(self) -> Dict[int, Any]:
        return self._X

    @property
    def y(self) -> Any:
        return self._y

    @property
    def samples(self) -> List[str]:
        return self._samples


class TokensRegressionProbingTask(ProbingTask):
    def __init__(
        self,
        name,
        get_target: Callable[[Sample], List[Union[int, float]]],
        description="",
    ):
        self.name = name
        self._get_target = get_target
        self.description = description.strip()

        self.r2id: Optional[dict] = None

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_target(self, sample: Sample):
        return self._get_target(sample)

    def get_probing_model(self) -> TokenRegressionLinearModel:
        return TokenRegressionLinearModel()

    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:
        logging.info("make dataset")

        def do(
            data: List[Sample], layers: List[int]
        ) -> Tuple[Dict[int, np.ndarray], Any, List[str]]:
            X_by_layer = defaultdict(list)
            y_list: List[Union[float, int]] = []
            samples: List[str] = []
            for elem in data:
                features = elem.features(handle="none")
                for layer in layers:
                    assert len(features) > layer
                    X_by_layer[layer].extend(features[layer].numpy())
                y_list.extend(self.get_target(elem))
                samples.extend(elem.bpe.split(" "))

            y_numpy = np.array(y_list)
            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer]
            return X_by_layer_numpy, y_numpy, samples

        X_train_by_layer, y_train, samples_train = do(train_data, layers)
        X_test_by_layer, y_test, samples_test = do(test_data, layers)
        return Dataset(X_train_by_layer, y_train, samples_train), Dataset(
            X_test_by_layer, y_test, samples_test
        )


def get_path_length(sample: Sample) -> List[Union[int, float]]:
    # length of the AST path from the root
    return [len(x.vec) for x in sample.representations]


def get_node_childcount(sample: Sample) -> List[Union[int, float]]:
    # number of children of the AST node
    return [len(x.node.children) for x in sample.nodes]


PROBINGS: List[ProbingTask] = []
PROBINGS.extend(
    [
        # TokensRegressionProbingTask(
        #     "Token Path Length",
        #     get_path_length,
        #     description="""
        #     Regression of the length of the path from the root to the node.
        # """,
        # ),
        # TokensRegressionProbingTask(
        #     "Token Child Count",
        #     get_node_childcount,
        #     description="""
        #     Regression of the number of children of the AST node.
        # """,
        # ),
    ]
)

if __name__ == "__main__":
    sample = Sample.default()
    for task in PROBINGS:
        print("Task:", task.get_name())
        print("Description:", task.get_description())
        print("Sample:", sample.bpe)
        print("Targets:", task.get_target(sample))
