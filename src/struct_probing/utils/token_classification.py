import logging
import random
import warnings
from collections import Counter, defaultdict
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Tuple,
                    Union)

import numpy as np
from src.struct_probing.probings.base import Metrics, Result
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from struct_probing.utils.tree_representation import NodeRepr
from src.struct_probing.utils.sample import Sample


def train_linear_model(X_train, y_train):
    model = RidgeClassifierCV(cv=4, alphas=[1e-3, 1e-2, 1e-1, 1], normalize=True)
    model.fit(np.array(X_train), np.array(y_train))
    return model


def predict_model(model, X_train):
    return model.predict(np.array(X_train))


def eval_prediction(y_true: Result, y_pred: Result) -> Metrics:
    return {
        "balanced_adj_acc": balanced_accuracy_score(y_true, y_pred, adjusted=True),
        "acc": accuracy_score(y_true, y_pred),
    }


class TokensClassificationProbingTask:
    def __init__(
        self, name, get_target: Callable[[Sample], List[Hashable]], description=""
    ):
        self.name = name
        self._get_target = get_target
        self.description = description.strip()

        self.r2id: Optional[dict] = None

    def make_dict(
        self, iterator: Iterable[Sample], max_classes, min_samples_per_class=30
    ) -> Dict[Hashable, int]:
        """preprocess a dict to enumerate classes

        Args:
            iterator (Iterable[Sample]): data samples
            max_classes ([type]): how many classes to preserve for classification
            min_samples_per_class (int, optional): If the class is less than min_samples_per_class, skip this class. Defaults to 30.

        Returns:
            [type]: [description]
        """
        # Hashable -> idx
        logging.info("make dict")
        r2id_counter = Counter(
            [elem for sample in iterator for elem in self._get_target(sample)]
        )
        logging.info(f"classes: {r2id_counter.most_common(n=max_classes)}")

        n_classes = min(
            max_classes, len(r2id_counter.values())
        )  # all other are mapped to OOV
        r2id = {}
        total = 0
        for i, elem in enumerate(r2id_counter.items()):
            total += elem[1]
        logging.info(f"total={total}")
        skipped = []
        for i, elem in enumerate(r2id_counter.items()):
            if elem[1] < min_samples_per_class:
                skipped.append(elem)
                r2id[elem[0]] = n_classes
                continue
            if i < n_classes:
                r2id[elem[0]] = i
            else:
                r2id[elem[0]] = n_classes  # OOV if any
        if len(skipped) > 0:
            warnings.warn(f"skipped classes by min_fraction_per_class: {skipped}")
        return r2id

    def get_target(self, x: List[NodeRepr], n_classes: int) -> List[int]:
        """get target for each token

        Args:
            x (List[NodeRepr]): list of node representations

        Raises:
            ValueError: if dict is not initialized

        Returns:
            List[int]: target for each token
        """
        if self.r2id is None:
            raise ValueError("r21d (dict relation 2 idx) is not initialized")

        def get(dct, elem):
            if not elem in dct:
                warnings.warn(f"elem {elem} not in r2id")
                return n_classes
            else:
                return dct[elem]

        return [get(self.r2id, elem) for elem in self._get_target(x)]

    def _make_dataset(
        self, data: List[Sample], layers: List[int], max_classes=30
    ) -> Tuple:
        max_samples = 2000
        logging.info(f"max samples for token classification is set to {max_samples}")
        data = data[:max_samples]
        logging.info("make dataset")
        self.r2id = self.make_dict(
            data, max_classes
        )  # preprocess targets to enumerate them

        random.seed(0)
        np.random.seed(0)
        train_data, test_data = train_test_split(data, train_size=0.8)

        def do(
            data: List[Sample], layers: List[int]
        ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
            X_by_layer = defaultdict(list)
            y_list: List[Union[float, int]] = []
            for elem in data:
                features = elem.features(handle="none")
                for layer in layers:
                    assert len(features) > layer
                    X_by_layer[layer].extend(features[layer].numpy())
                y_list.extend(self.get_target(elem, max_classes))

            y_numpy = np.array(y_list)
            where = y_numpy < max_classes  # filter OOV
            y_numpy = y_numpy[where]
            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer][where]
            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return X_by_layer_numpy, y_numpy

        X_train_by_layer, y_train = do(train_data, layers)
        X_test_by_layer, y_test = do(test_data, layers)
        return X_train_by_layer, y_train, X_test_by_layer, y_test

    def _probe(self, dataset: Tuple) -> Dict[int, Tuple[Result, Result]]:
        X_train_by_layer, y_train, X_test_by_layer, y_test = dataset
        result_by_layer = {}
        for layer in X_train_by_layer.keys():
            X_train = X_train_by_layer[layer]
            X_test = X_test_by_layer[layer]
            model = train_linear_model(X_train, y_train)

            train_pred = predict_model(model, X_train)
            test_pred = predict_model(model, X_test)

            train = eval_prediction(y_train, train_pred)
            test = eval_prediction(y_test, test_pred)

            result_by_layer[layer] = (
                Result("train", train, y_train, train_pred),
                Result("test", test, y_test, test_pred),
            )
        return result_by_layer


def get_path_class(sample: Sample) -> List[Hashable]:
    # type of path for the AST path from the root
    return [str(x) for x in sample.representations]


def get_path_length(sample: Sample) -> List[Hashable]:
    # length of the AST path from the root
    return [len(x.vec) for x in sample.representations]


def get_node_type(sample: Sample) -> List[Hashable]:
    # type of the AST node
    assert len(list(sample.nodes)) == len(
        list(sample.representations)
    ), f"{len(sample.representations)}, {len(sample.bpe.strip().split())}, {len(list(sample.nodes))}"
    return [str(x.node.type) for x in sample.nodes]


def get_node_childcount(sample: Sample) -> List[Hashable]:
    # number of children of the AST node
    return [str(x.node.child_count) for x in sample.nodes]


PROBINGS = []
PROBINGS.extend(
    [
        TokensClassificationProbingTask(
            "Token Path Type",
            get_path_class,
            description="""
            Classification of types of paths from the root to the node.
        """,
        ),
        TokensClassificationProbingTask(
            "Token Path Length",
            get_path_length,
            description="""
            Classification of the length of the path from the root to the node.
        """,
        ),
    ]
)
PROBINGS.extend(
    [
        TokensClassificationProbingTask(
            "Token Node Type",
            get_node_type,
            description="""
            Classification of the type of AST node
        """,
        ),
    ]
)

if __name__ == "__main__":
    sample = Sample.default()
    for task in PROBINGS:
        print("Task:", task.name)
        print("Description:", task.description)
        print("Sample:", sample.bpe)
        print("Targets:", task._get_target(sample))
