import logging
import random
import warnings
from collections import Counter, defaultdict
from time import sleep
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Tuple, Union)

import numpy as np
from src.struct_probing.probings.base import ProbingDataset, ProbingModel, ProbingTask
# from sklearn.neural_network import MLPClassifier
from src.struct_probing.probings.mlp_utils import TorchMLPClassifier
from src.struct_probing.probings.tfidf_utils import TfIdfClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.struct_probing.utils.sample import Sample

# log = logging.Logger("sentence_regression", level=logging.INFO)


class SentenceClassificationLinearModel(ProbingModel):
    def __init__(self):
        pass

    def train_linear_model(self, X_train, y_train):
        model = LogisticRegressionCV(
            Cs=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100),
            random_state=777,
            max_iter=10000,
        )
        model.fit(np.array(X_train), np.array(y_train))
        logging.info(str(model))
        return model

    def predict_linear_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def train_upper_bound_model(self, X_train, y_train):
        model = TorchMLPClassifier()
        model.fit(np.array(X_train), np.array(y_train))
        return model

    def predict_upper_bound_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def eval_prediction(self, y_true: Any, y_pred: Any):
        return {
            "1 - balanced_adj_acc": 1
            - balanced_accuracy_score(y_true, y_pred, adjusted=True),
            "1 - acc": 1 - accuracy_score(y_true, y_pred),
        }

    def train_lower_bound(
        self, samples: List[str], y_train: List[Any]
    ) -> Callable[[str], Any]:
        return lambda _sample: Counter(y_train).most_common(n=1)[0][0]

    def train_bag_of_words(
        self,
        samples: List[Sample],
        y_train: List[Any],
        sample2hashes: Callable[[Sample], List[Hashable]],
    ) -> Any:
        model = TfIdfClassifier()
        X_train = [sample2hashes(sample) for sample in samples]
        model.fit(X_train, y_train)
        return model

    def predict_bag_of_words(
        self,
        model: Any,
        samples: List[Sample],
        sample2hashes: Callable[[Sample], List[Hashable]],
    ) -> Any:
        X_test = [sample2hashes(sample) for sample in samples]
        return model.predict(X_test)


class Dataset(ProbingDataset):
    def __init__(
        self, X: Dict[int, Any], y: List[Any], samples: List[Sample], y_raw=None
    ):
        self._X = X
        self._y = y
        self._samples = samples
        self.y_raw = y_raw

    @property
    def X_by_layer(self) -> Dict[int, Any]:
        return self._X

    @property
    def y(self) -> Any:
        return self._y

    @property
    def samples(self) -> List[Sample]:
        return self._samples


class SentenceClassificationProbingTask(ProbingTask):
    def __init__(
        self,
        name: str,
        get_target: Callable[[Sample], Hashable],
        code_aug_type="identity",
        dataset_type="mlm",
        embedding_handle="mean",
        embedding_type="dummy",
        description="",
        max_classes=30,
        skip_label=None,
        sample2hashable=None,
    ):
        self.name = name
        self.description = description.strip()
        self._get_target = get_target
        self.code_aug_type = code_aug_type
        self.dataset_type = dataset_type

        self.r2id: Optional[dict] = None
        self.max_classes = max_classes
        self.skip_label = skip_label
        self.embedding_type = embedding_type
        self.embedding_handle = embedding_handle
        self._sample2hashable = sample2hashable

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_target(self, x: Sample) -> Hashable:
        return self._get_target(x)

    def get_probing_model(self) -> SentenceClassificationLinearModel:
        return SentenceClassificationLinearModel()

    def get_augmentation(self) -> str:
        return self.code_aug_type

    def get_dataset(self) -> str:
        return self.dataset_type

    def get_embedding_type(self) -> str:
        return self.embedding_type

    def get_sample2hashable(self):
        # return self._sample2hashable(sample)
        return lambda sample: sample.bpe

    def make_dict(
        self, iterator: Iterable[Sample], max_classes, min_samples_per_class=1
    ) -> Dict[Hashable, int]:
        """preprocess a dict to enumerate classes

        Args:
            iterator (Iterable[Sample]): data samples
            max_classes ([type]): how many classes to preserve for classification
            min_samples_per_class (int, optional): If the class is less than min_samples_per_class, skip this class. Defaults to 30.

        Returns:
            Dict[Hashable, int]: a dictionary class -> label
        """
        # Hashable -> idx
        logging.info("make dict")
        targets = [
            self._get_target(sample) for sample in iterator
        ]  # a single tartet per sentence

        if self.skip_label is not None:
            to_skip = self.skip_label
            if not isinstance(to_skip, list):
                to_skip = [self.skip_label]
            for skip in to_skip:
                warnings.warn(f"skip labels: {skip}")
                targets = list(filter(lambda x: x != skip, targets))
        r2id_counter = Counter(targets)

        logging.info(f"classes: {r2id_counter.most_common(n=max_classes)}")

        n_classes = min(
            max_classes, len(r2id_counter.values())
        )  # all other are mapped to OOV
        r2id = {}
        total = 0
        for i, elem in enumerate(r2id_counter.most_common()):
            total += elem[1]
        logging.info(f"total={total}")
        skipped = []
        class_index = 0
        for i, elem in enumerate(r2id_counter.most_common()):
            if elem[1] < min_samples_per_class:
                skipped.append(elem)
                r2id[elem[0]] = n_classes
                continue
            else:
                logging.info(f"added: {elem}")

            r2id[elem[0]] = class_index
            class_index += 1
        # if len(skipped) > 0:
        #     warnings.warn(
        #         f"skipped classes by min_fraction_per_class: {len(skipped)}: {skipped[0]}.."
        #     )
        return r2id

    def _map_target(self, elem: Hashable) -> int:
        if self.r2id is None:
            raise ValueError("r21d (dict relation 2 idx) is not initialized")

        def get(dct, elem):
            if not elem in dct:
                # warnings.warn(f"elem {elem} not in r2id")
                return self.max_classes
            else:
                return dct[elem]

        return get(self.r2id, elem)

    def _make_dataset(
        self,
        train_data: List[Sample],
        test_data: List[Sample],
        layers: List[int],
        **kwargs,
    ) -> Tuple[Dataset, Dataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(train_data, self.max_classes)

        def do(data: List[Sample], layers):
            X_by_layer = defaultdict(list)
            y_list = []
            y_list_raw = []
            samples = []
            for data_sample in data:
                features = data_sample.features(handle=self.embedding_handle)

                for layer in layers:
                    assert len(features) > layer
                    X_by_layer[layer].append(features[layer].numpy())
                y_list.append(self._map_target(self.get_target(data_sample)))
                y_list_raw.append(self.get_target(data_sample))
                samples.append(data_sample)

            y_numpy = np.array(y_list)
            y_numpy_raw = np.array(y_list_raw)
            where = y_numpy < self.max_classes  # filter OOV
            y_numpy = y_numpy[where]
            y_numpy_raw = y_numpy_raw[where]
            samples = [samples[i] for i in np.arange(len(samples))[where]]
            logging.info(f"remained y_numpy: {str(np.unique(y_numpy))}")
            logging.info(f"remained y_numpy_raw: {str(np.unique(y_numpy_raw))}")
            sleep(2)

            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer][where]

            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return X_by_layer_numpy, y_numpy, y_numpy_raw, samples

        X_train_by_layer, y_train, y_train_raw, train_samples = do(train_data, layers)
        X_test_by_layer, y_test, y_test_raw, test_samples = do(test_data, layers)
        assert len(X_train_by_layer[0]) == len(y_train), (
            len(X_train_by_layer[0]),
            len(y_train),
        )
        return Dataset(
            X_train_by_layer, y_train, train_samples, y_raw=y_train_raw
        ), Dataset(X_test_by_layer, y_test, test_samples, y_raw=y_test_raw)


def get_readability(sample: Sample) -> Hashable:
    return int(float(sample.default_target) > 3.6)


def get_func_name(sample: Sample) -> Hashable:
    return sample.sentence_label


def get_sort(sample: Sample) -> Hashable:
    return sample.default_target


def token_is_var_misused(sample: Sample) -> Hashable:
    assert all([sample.bpe_aug_labels[0] == label for label in sample.bpe_aug_labels])
    bpe_aug_labels: Hashable = sample.bpe_aug_labels[0]
    return bpe_aug_labels


PROBINGS = [
    SentenceClassificationProbingTask(
        "readability",
        get_readability,
        code_aug_type="readability",
        dataset_type="readability",
        embedding_handle="ident",
        embedding_type="mean",
        description="""
            Predicts readability of the sentence
        """,
    ),
    # SentenceClassificationProbingTask(
    #     "Function Name",
    #     get_func_name,
    #     code_aug_type="funcname",
    #     dataset_type="funcname",
    #     embedding_handle="ident",
    #     embedding_type="mean",
    #     description="""
    #         Predicts the function name
    #     """,
    # ),
    SentenceClassificationProbingTask(
        "Algorithm",
        get_sort,
        code_aug_type="algo",
        dataset_type="algo_1671-A",
        embedding_handle="ident",
        embedding_type="mean",
        description="""
            Predicts if the algo is wrong/correct
        """,
    ),
    SentenceClassificationProbingTask(
        "Is Variable Misused",
        token_is_var_misused,
        code_aug_type="varmisuse",
        description="""
        binary classification: is variable misused (swapped with other) or original
        """,
        embedding_handle="mean",
    ),
]
