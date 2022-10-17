import logging
import random
from collections import Counter, defaultdict
from turtle import clone
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from src.struct_probing.probings.base import (Metrics, ProbingDataset, ProbingModel,  # Result
                           ProbingTask)
from sklearn.linear_model import \
    SGDClassifier  # LogisticRegressionCV, RidgeClassifierCV,
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from src.struct_probing.utils.sample import Sample

log = logging.Logger("token_classification", level=logging.INFO)


class Labels:
    Original = 0
    SemanticPreserving = 1
    NonSemanticPreserving = 2


class TokenLabels:
    Original = 0
    New = 1
    RelevantOriginal = 2


class LowerBoundModel:
    def __init__(self, bpe2most_common: Dict[str, Any], global_mostcommon: dict):
        self.bpe2most_common = bpe2most_common
        self.global_mostcommon = global_mostcommon

    def __call__(self, bpe: str) -> Any:
        if bpe in self.bpe2most_common:
            return self.bpe2most_common[bpe]
        else:
            return self.global_mostcommon  # handle OOV bpe token


class TokenClassificationLinearModel(ProbingModel):
    def __init__(self):
        pass

    def train_linear_model(self, X_train, y_train):
        # model = LogisticRegressionCV(multi_class="multinomial", max_iter=50)
        params = {"alpha": [0.0001, 0.001, 0.01, 0.1]}
        model = SGDClassifier(loss="log", verbose=0, tol=0.0001)
        grid = GridSearchCV(model, param_grid=params, verbose=3)
        grid.fit(np.array(X_train), np.array(y_train))
        return grid

    def predict_linear_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def train_upper_bound_model(self, X_train, y_train):
        model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), verbose=True)
        model.fit(np.array(X_train), np.array(y_train))
        return model

    def predict_upper_bound_model(self, model, X_train):
        return model.predict(np.array(X_train))

    def eval_prediction(self, y_true: Any, y_pred: Any) -> Metrics:
        return {
            "1 - balanced_adj_acc": 1
            - balanced_accuracy_score(y_true, y_pred, adjusted=True),
            "1 - acc": 1 - accuracy_score(y_true, y_pred),
        }

    def train_lower_bound(
        self, samples: List[str], y_train: List[Any]
    ) -> Callable[[str], Any]:
        assert len(samples) == len(y_train)
        bpe2labels: Dict[str, Counter] = defaultdict(Counter)
        for bpe, label in zip(samples, y_train):
            bpe2labels[bpe].update([label])
        global_mostcommon = Counter(y_train).most_common(1)[0][0]
        bpe2most_common = {
            bpe: counter.most_common(1)[0][0] for bpe, counter in bpe2labels.items()
        }
        model = LowerBoundModel(bpe2most_common, global_mostcommon)
        return model


class Dataset(ProbingDataset):
    def __init__(self, X: Dict[int, Any], y: np.ndarray, samples: List[Sample]):
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


class CloneDetectionProbingTask(ProbingTask):
    def __init__(
        self,
        name,
        get_target: Callable[[Sample], List[Hashable]],
        skip_label=None,
        code_aug_type="identity",
        description="",
        max_classes=15,
    ):
        self.name = name
        self._get_target = get_target
        self.skip_label = skip_label
        self.description = description.strip()
        self.code_aug_type = code_aug_type

        self.r2id: Optional[dict] = None
        self.max_classes = max_classes

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_augmentation(self) -> str:
        return self.code_aug_type  # can be changed

    def get_target(self, x: Sample) -> List[Hashable]:
        return self._get_target(x)

    def get_probing_model(self) -> TokenClassificationLinearModel:
        return TokenClassificationLinearModel()

    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:

        orig_by_sent_id_train = defaultdict(list)
        orig_by_sent_id_test = defaultdict(list)

        clone_by_sent_id = defaultdict(list)

        for elem in train_data:
            if elem.sentence_label == Labels.Original:
                orig_by_sent_id_train[elem.sentence_id].append(elem)

        for elem in test_data:
            if elem.sentence_label == Labels.Original:
                orig_by_sent_id_test[elem.sentence_id].append(elem)

        # todo separate!
        for elem in train_data + test_data:
            if elem.sentence_label != Labels.Original:
                clone_by_sent_id[elem.sentence_id].append(elem)

        assert (
            len(
                set(orig_by_sent_id_train.keys()).union(
                    set(orig_by_sent_id_test.keys())
                )
            )
            == 0
        )
        assert set(orig_by_sent_id_train.keys()).union(
            set(orig_by_sent_id_test.keys())
        ) == set(clone_by_sent_id.keys())

        def get_pairs(orig_dict: Dict[int, List[Any]]):
            pairs = []
            for sent_id in orig_dict.keys():
                for orig in orig_dict[sent_id]:
                    for clone in clone_by_sent_id[sent_id]:
                        pairs.append((orig, clone))
            return pairs

        train_pairs, test_pairs = get_pairs(orig_by_sent_id_train), get_pairs(
            orig_by_sent_id_test
        )

        def do(data: List[Tuple[Sample, Sample]], layers: List[int]) -> Dataset:
            X_by_layer = defaultdict(list)
            y_list = []
            for orig, clone in data:
                ind_orig = []
                for i, (label, bpe) in enumerate(
                    orig.bpe_aug_labels, orig.bpe.split(" ")
                ):
                    if label == TokenLabels.RelevantOriginal:
                        print("relevant old", bpe)
                        ind_orig.append(i)

                ind_new = []
                for i, (label, bpe) in enumerate(
                    clone.bpe_aug_labels, clone.bpe.split(" ")
                ):
                    if label == TokenLabels.New:
                        print("new", bpe)
                        ind_new.append(i)

                features_orig = orig.features(handle="none")
                features_clone = clone.features(handle="none")
                for layer in layers:
                    assert len(features_orig) > layer

                    gathered_orig = torch.cat(
                        [features_orig[layer][i].unsqueeze(0) for i in ind_orig], dim=0
                    )
                    mean_orig = gathered_orig.mean(0)

                    gathered_clone = torch.cat(
                        [features_clone[layer][i].unsqueeze(0) for i in ind_new], dim=0
                    )
                    mean_clone = gathered_clone.mean(0)

                    concat_features = np.concatenate(
                        (mean_orig.numpy(), mean_clone.numpy()),
                        axis=-1,
                    )
                    print(concat_features.shape)
                    X_by_layer[layer].append(concat_features)
                y_list.append(self.get_target(clone))

            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
            y_numpy = np.array(y_list)

            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return Dataset(X_by_layer_numpy, y_numpy, [it[1] for it in data])

        return do(train_pairs, layers), do(test_pairs, layers)


def get_sentence_label(sample_clone: Sample) -> List[Hashable]:
    assert sample_clone.sentence_label in {
        Labels.SemanticPreserving,
        Labels.NonSemanticPreserving,
    }
    return sample_clone.sentence_label


PROBINGS: List[CloneDetectionProbingTask] = []
PROBINGS.extend(
    [
        CloneDetectionProbingTask(
            "Is Semantic Clone",
            get_sentence_label,
            skip_label=0,
            code_aug_type="variable_insert",
            description="""
            Binary Classification: is variable inside print statement not declarated
            `0` label is not used
        """,
        ),
    ]
)

if __name__ == "__main__":
    sample = Sample.default()
    for task in PROBINGS:
        print("Task:", task.get_name())
        print("Description:", task.get_description())
        print("Sample:", sample.bpe)
        print("Targets:", task._get_target(sample))
