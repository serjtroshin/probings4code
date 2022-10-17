import logging
import pickle
import torch
import random
import warnings
from collections import Counter, defaultdict
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Tuple, Union)

import numpy as np
import src.struct_probing.probings.edge_utils as edge_utils

from src.struct_probing.probings.base import ProbingDataset, ProbingModel, ProbingTask
from sklearn.linear_model import RidgeCV, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GridSearchCV

from src.struct_probing.utils.sample import Sample
from src.struct_probing.utils.tree_representation import NodeRepr

from .mlp_utils import TorchMLPClassifier, TorchMLPRegressor
from .token_classification import LowerBoundModel

log = logging.Logger("token_regression", level=logging.INFO)


class EdgeClassificationLinearModel(ProbingModel):
    def __init__(self):
        params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
        model = SGDClassifier(loss="log", verbose=0, tol=0.0001)
        grid = GridSearchCV(model, param_grid=params, verbose=3)

        self.model = edge_utils.SimpleModel(grid, emb_dim=768)

    def train_linear_model(self, X_train, y_train):
        h_i = np.vstack([x[0] for x in X_train])
        h_j = np.vstack([x[1] for x in X_train])
        self.model.fit(h_i, h_j, np.array(y_train))
        return self.model

    def predict_linear_model(self, model, X_train):
        h_i = np.vstack([x[0] for x in X_train])
        h_j = np.vstack([x[1] for x in X_train])
        return model.predict(h_i, h_j)

    def train_upper_bound_model(self, X_train, y_train):
        model = TorchMLPClassifier()
        h_i = np.vstack([x[0] for x in X_train])
        h_j = np.vstack([x[1] for x in X_train])
        h = np.hstack([h_i, h_j])
        model.fit(h, np.array(y_train))
        return model

    def predict_upper_bound_model(self, model, X_train):
        h_i = np.vstack([x[0] for x in X_train])
        h_j = np.vstack([x[1] for x in X_train])
        h = np.hstack([h_i, h_j])
        return model.predict(h)

    def eval_prediction(self, y_true, y_pred) -> dict:
        # assert y_true.shape == y_pred.shape
        return {
            "1 - balanced_adj_acc": 1
            - balanced_accuracy_score(y_true, y_pred, adjusted=True),
            "1 - acc": 1 - accuracy_score(y_true, y_pred),
        }

    # def train_lower_bound(
    #     self, samples: List[Tuple[str, str]], y_train: List[Any]
    # ) -> Callable[[str], Any]:

    #     bpe2labels: Dict[str, list] = defaultdict(list)
    #     for bpe, label in zip(samples, y_train):
    #         bpe2labels[bpe].append(label)
    #     global_mean = np.median(y_train)
    #     bpe2mean = {bpe: np.median(values) for bpe, values in bpe2labels.items()}

    #     def model(bpe: str) -> Any:
    #         if bpe in bpe2mean:
    #             return bpe2mean[bpe]
    #         else:
    #             return global_mean  # handle OOV bpe token
    #     return model

    def train_lower_bound(
        self, samples: List[Tuple[str, str]], y_train: List[Any]
    ) -> Callable[[str], Any]:

        assert len(samples) == len(y_train)
        bpe2labels: Dict[str, Counter] = defaultdict(Counter)
        for bpe, label in zip(samples, y_train):
            # print(bpe)
            # input()
            bpe2labels[bpe].update([label])
        global_mostcommon = Counter(y_train).most_common(1)[0][0]
        bpe2most_common = {
            bpe: counter.most_common(1)[0][0] for bpe, counter in bpe2labels.items()
        }
        model = LowerBoundModel(bpe2most_common, global_mostcommon)
        print(bpe2most_common)
        return model

        # return lambda _sample: Counter(y_train).most_common(n=1)[0][0]


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


class EdgeClassificationProbingTask(ProbingTask):
    def __init__(
        self,
        name,
        get_target: Callable[[Sample], List[Tuple[int, int, Any]]],
        description="",
        code_aug_type="identity",
    ):
        self.name = name
        self._get_target = get_target
        self.description = description.strip()
        self.code_aug_type = code_aug_type

        self.r2id: Optional[dict] = None
        self.max_classes = 10000000  # inf
        self.skip_label = None

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_target(self, sample: Sample) -> List[Tuple[int, int, Any]]:
        """get edges targets

        Returns:
            List[Tuple[int, int, Any]]: list of
            (h_i's token_id, h_j's token_id, target(w_i, w_j))
        """
        return self._get_target(sample)

    def get_augmentation(self) -> str:
        return self.code_aug_type  # can be changed

    def get_probing_model(self) -> EdgeClassificationLinearModel:
        return EdgeClassificationLinearModel()

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
            label for sample in iterator for _, _, label in self._get_target(sample)
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

    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(train_data, self.max_classes)

        def do(
            data: List[Sample], layers: List[int]
        ) -> Tuple[Dict[int, np.ndarray], Any, List[str]]:
            h_i_by_layer = defaultdict(list)
            h_j_by_layer = defaultdict(list)
            y_list: List[Union[float, int]] = []
            samples: List[Tuple[str, str]] = []  # bpe_i bpe_j

            #              # resamples
            #             min_class_num = 10000000
            #             for label in np.array(y_numpy):
            #                 where = y_numpy == label
            #                 min_class_num = min(min_class_num, len(where))
            #             logging.info(f"min_class_num: {min_class_num}")

            #             all_where = np.zeros(y_numpy.shape, dtype=np.bool_)
            #             for label in np.array(y_numpy):
            #                 where = y_numpy == label
            #                 where = where[:min_class_num]
            #                 all_where = all_where | where
            #             logging.info(f"remained total: {np.sum(all_where)}")
            #             input()

            selected = dict()
            for label in self.r2id.keys():
                selected[label] = list()

            for id_, elem in enumerate(data):
                bpes = elem.bpe.split(" ")
                targets = self.get_target(elem)

                for i, j, label in targets:
                    selected[label].append((id_, i, j, label))
            #             random.seed(0)
            #             for label in selected:
            #                 random.shuffle(selected[label])
            min_cnt = min(map(lambda x: len(x), selected.values()))
            logging.info(f"min targets for labels: {min_cnt}")
            for label in selected:
                selected[label] = selected[label][:min_cnt]

            selected_by_id = defaultdict(list)
            for label in selected:
                for id_, i, j, _ in selected[label]:
                    selected_by_id[id_].append((i, j, label))

            for id_, elem in enumerate(data):
                bpes = elem.bpe.split(" ")
                #                 targets = self.get_target(elem)
                features = elem.features(handle="none")
                for i, j, label in selected_by_id[id_]:
                    for layer in layers:
                        assert len(features) > layer

                        def get_mean_emb(embs, start, end):
                            return torch.cat(
                                [emb.unsqueeze(0) for emb in embs[start : end + 1]],
                                axis=0,
                            ).mean(0)

                        h_i_by_layer[layer].append(
                            get_mean_emb(features[layer], *i).numpy()
                        )
                        h_j_by_layer[layer].append(
                            get_mean_emb(features[layer], *j).numpy()
                        )

                    def get_bpe(bpes, start, end):
                        return " ".join(bpes[start : end + 1])

                    samples.append((get_bpe(bpes, *i), get_bpe(bpes, *j)))
                    y_list.append(self._map_target(label))
            # logging.info("unique targets:", str(y_numpy.unique()))

            logging.info(f"targets: {Counter(y_list).most_common()}")

            y_numpy = np.array(y_list)  # [:, None]
            print("target shape", y_numpy.shape)

            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(
                    list(zip(h_i_by_layer[layer], h_j_by_layer[layer]))
                )

            return X_by_layer_numpy, y_numpy, samples

        X_train_by_layer, y_train, samples_train = do(train_data, layers)
        X_test_by_layer, y_test, samples_test = do(test_data, layers)
        return Dataset(X_train_by_layer, y_train, samples_train), Dataset(
            X_test_by_layer, y_test, samples_test
        )


def get_data_flow(
    sample: Sample,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Union[int, float]]]:
    class Labels:
        NoEdge = "NoEdge"
        comesFrom = "comesFrom"
        computedFrom = "computedFrom"

    # distance between the nodes in AST tree
    edges = sample.edge_labels
    edges = [
        (obj.first, obj.second, obj.label) for obj in edges
    ]  # pairs of start id, end id of dfg edge, and label

    # positive_edges = [edge for edge in edges if edge[-1] != Labels.NoEdge]
    # negative_edges = [edge for edge in edges if edge[-1] == Labels.NoEdge]
    # idx = np.random.choice(
    #     np.arange(len(negative_edges)),
    #     size=min(len(negative_edges), len(positive_edges)),
    #     replace=False
    # )
    # selected_edges = positive_edges + [negative_edges[i] for i in idx]
    return edges


def get_ast_dist(sample: Sample) -> List[Tuple[int, int, Union[int, float]]]:
    # distance between the nodes in AST tree
    n = len(sample.nodes)
    targets = []
    all_pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    all_pairs = all_pairs[np.random.choice(np.arange(len(all_pairs)), size=n)]
    for i, j in all_pairs:
        dist = NodeRepr(sample.nodes[i].representation).dist(
            NodeRepr(sample.nodes[j].representation)
        )
        targets.append((i, j, dist))
    return targets


PROBINGS: List[ProbingTask] = []
PROBINGS.extend(
    [
        # EdgeRegressionProbingTask(
        #     "AST distance",
        #     get_ast_dist,
        #     description="""
        #     Regression of the length of the path between two nodes.
        # """,
        # ),
        EdgeClassificationProbingTask(
            "DFG",
            get_data_flow,
            code_aug_type="dfg",
            description="""
            Classification if the edge in dfg.
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
        print("Targets:", task.get_target(sample))
