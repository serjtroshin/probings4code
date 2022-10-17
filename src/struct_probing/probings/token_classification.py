import logging
import random
import string
import warnings
from collections import Counter, defaultdict
from time import sleep
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Tuple, Union)

import numpy as np
import torch
from src.struct_probing.probings.base import (Metrics, ProbingDataset, ProbingModel,
                           ProbingNgramDataset, ProbingTask)
# from sklearn.neural_network import MLPClassifier
from src.struct_probing.probings.mlp_utils import TorchMLPClassifier
from src.struct_probing.probings.tfidf_utils import TfIdfClassifier
from sklearn.linear_model import \
    SGDClassifier  # LogisticRegressionCV, RidgeClassifierCV,
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.struct_probing.utils.sample import Sample

log = logging.Logger("token_classification", level=logging.INFO)


class LowerBoundModel:
    def __init__(self, bpe2most_common: Dict[str, Any], global_mostcommon: dict):
        self.bpe2most_common = bpe2most_common
        self.global_mostcommon = global_mostcommon

    def __call__(self, bpe: str) -> Any:
        if bpe in self.bpe2most_common:
            return self.bpe2most_common[bpe]
        else:
            return self.global_mostcommon  # handle OOV bpe token


class LowerBoundNgrammModel:
    def __init__(
        self, bpe2most_common: Dict[str, Any], global_mostcommon: dict, n: int
    ):
        self.bpe2most_common = bpe2most_common
        self.global_mostcommon = global_mostcommon
        self.n = n

    def __call__(self, data: ProbingNgramDataset) -> List[Any]:
        preds = []
        context_str = data.context_str(n=self.n)
        for context in context_str:
            if context in self.bpe2most_common:
                preds.append(self.bpe2most_common[context])
            else:
                preds.append(self.global_mostcommon)  # handle OOV bpe token
        return preds


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
        # model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), verbose=True)
        model = TorchMLPClassifier()
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
        samples = ["" for _ in range(len(samples))]
        assert len(samples) == len(y_train)
        bpe2labels: Dict[str, Counter] = defaultdict(Counter)
        for bpe, label in zip(samples, y_train):
            bpe2labels[bpe].update([label])
        global_mostcommon = Counter(y_train).most_common(1)[0][0]
        bpe2most_common = {
            bpe: counter.most_common(1)[0][0] for bpe, counter in bpe2labels.items()
        }
        model = LowerBoundModel(bpe2most_common, global_mostcommon)
        print(bpe2most_common)
        return model

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

    def train_n_gramm_lower_bound(
        self, data: ProbingNgramDataset, n=3
    ) -> Callable[[ProbingNgramDataset], Any]:

        bpe2labels: Dict[str, Counter] = defaultdict(Counter)
        context_str = data.context_str(n=n)
        assert len(context_str) == len(data.y)
        for context_str, label in zip(context_str, data.y):
            bpe2labels[context_str].update([label])
        global_mostcommon = Counter(data.y).most_common(1)[0][0]
        bpe2most_common = {
            bpe: counter.most_common(1)[0][0] for bpe, counter in bpe2labels.items()
        }
        model = LowerBoundNgrammModel(bpe2most_common, global_mostcommon, n=n)
        return model


class NgramDataset(ProbingNgramDataset):
    def __init__(
        self, bpes: List[List[str]], samples: List[Tuple[int, int]], y: np.ndarray
    ):
        assert len(samples) == len(y)
        self._bpes = bpes
        self._y = y
        self._samples = samples

    @property
    def bpes(self) -> List[List[str]]:
        return self._bpes

    @property
    def positions(self) -> List[Tuple[int, int]]:
        return self._samples

    @property
    def y(self) -> Any:
        return self._y


class Dataset(ProbingDataset):
    def __init__(
        self,
        X: Dict[int, Any],
        y: np.ndarray,
        samples: List[str],
        node_types: List[str] = None,
        all_bpes: List[List[str]] = None,
        all_node_types: List[List[str]] = None,
        token2id: List[Tuple[int, int]] = None,
        sent_ids=None,
    ):
        self._X = X
        self._y = y
        self._samples = samples  # list of bpe tokens
        # additional information
        self.node_types = node_types
        self.all_bpes = all_bpes
        self.all_node_types = all_node_types
        self.token2id = token2id
        self._sent_ids = sent_ids

    @property
    def X_by_layer(self) -> Dict[int, Any]:
        return self._X

    @property
    def y(self) -> Any:
        return self._y

    @property
    def samples(self) -> List[str]:
        return self._samples


class TokensClassificationProbingTask(ProbingTask):
    def __init__(
        self,
        name,
        get_target: Callable[[Sample], List[Hashable]],
        code_aug_type="identity",
        description="",
        max_classes=15,
        skip_label: Optional[Any] = None,
        embedding_handle="none",
    ):
        self.name = name
        self._get_target = get_target
        self.skip_label = skip_label
        self.description = description.strip()
        self.code_aug_type = code_aug_type
        self.embedding_handle = embedding_handle

        self.r2id: Optional[dict] = None
        self.max_classes = max_classes

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_augmentation(self) -> str:
        return self.code_aug_type  # can be changed

    def get_sample2hashable(self):
        # return self._sample2hashable(sample)
        return lambda sample: sample.bpe

    def make_dict(
        self, iterator: Iterable[Sample], max_classes, min_samples_per_class=30
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
        targets = [elem for sample in iterator for elem in self._get_target(sample)]

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
        logging.info(f"max classes: {n_classes}")
        r2id = {}
        total = 0
        for i, elem in enumerate(r2id_counter.items()):
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
        if len(skipped) > 0:
            warnings.warn(
                f"skipped classes by min_fraction_per_class: {len(skipped)}: {skipped[0]}.."
            )
        return r2id

    def get_target(self, x: List[Sample]) -> List[Hashable]:
        return self._get_target(x)

    def get_probing_model(self) -> TokenClassificationLinearModel:
        return TokenClassificationLinearModel()

    def _map_target(self, targets: List[Hashable]) -> List[int]:
        """get target for each token

        Args:
            targets (List[Hashable]): list of targets

        Raises:
            ValueError: if dict is not initialized

        Returns:
            List[int]: target for each token
        """
        if self.r2id is None:
            raise ValueError("r21d (dict relation 2 idx) is not initialized")

        def get(dct, elem):
            if not elem in dct:
                # warnings.warn(f"elem {elem} not in r2id")
                return self.max_classes
            else:
                return dct[elem]

        return [get(self.r2id, elem) for elem in targets]

    def _make_ngram_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingNgramDataset, ProbingNgramDataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(
            train_data, self.max_classes
        )  # preprocess targets to enumerate them

        def do(data: List[Sample], layers: List[int]) -> Dataset:
            y_list: List[Union[float, int]] = []
            samples: List[int] = []
            bpes: List[List[str]] = []

            for id_, elem in enumerate(data):
                targets = self._get_target(elem)
                y_list.extend(self._map_target(targets))
                bpe = elem.bpe.split(" ")
                bpes.append(bpe)
                samples.extend([(id_, j) for j in range(len(bpe))])

            y_numpy = np.array(y_list)
            samples_numpy = np.array(samples)
            where = y_numpy < self.max_classes  # filter OOV
            y_numpy = y_numpy[where]
            samples_numpy = samples_numpy[where]

            logging.info(f"remained y_numpy: {str(np.unique(y_numpy))}")
            sleep(2)

            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return NgramDataset(bpes=bpes, samples=samples_numpy, y=y_numpy)

        return do(train_data, layers), do(test_data, layers)

    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(
            train_data, self.max_classes
        )  # preprocess targets to enumerate them

        def do(data: List[Sample], layers: List[int]) -> Dataset:
            X_by_layer = defaultdict(list)
            y_list: List[Union[float, int]] = []
            samples: List[str] = []
            node_types: List[str] = []
            token2id = []
            all_bpes = []
            all_node_types = []
            sent_ids = []
            for id_, elem in enumerate(data):
                features = elem.features(handle=self.embedding_handle)
                for layer in layers:
                    assert len(features) > layer
                    X_by_layer[layer].extend(features[layer].numpy())
                targets = self._get_target(elem)
                targets = self._map_target(targets)

                sent_ids.extend([elem.sentence_id] * len(targets))
                y_list.extend(targets)

                samples.extend(elem.bpe.split(" "))
                node_types.extend([node.node.type for node in elem.nodes])
                token2id.extend([(id_, j) for j in range(len(elem.nodes))])
                all_bpes.append(elem.bpe.split(" "))
                all_node_types.append([x.node.type for x in elem.nodes])

            y_numpy = np.array(y_list)
            logging.info(f"unique y_numpy: {str(np.unique(y_numpy))}")
            where = y_numpy < self.max_classes  # filter OOV
            y_numpy = y_numpy[where]
            logging.info(f"remained y_numpy: {str(np.unique(y_numpy))}")
            sleep(2)

            samples_numpy = np.array(samples)[where]
            node_types_numpy = np.array(node_types)[where]
            sent_ids = np.array(sent_ids)[where]
            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer][where]
            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return Dataset(
                X_by_layer_numpy,
                y_numpy,
                samples_numpy,
                node_types_numpy,
                all_bpes,
                all_node_types,
                token2id,
                sent_ids=sent_ids,
            )

        return do(train_data, layers), do(test_data, layers)


def get_path_class(sample: Sample) -> List[Hashable]:
    # type of path for the AST path from the root
    return [str(x) for x in sample.representations]


def get_node_type(sample: Sample) -> List[Hashable]:
    # type of the AST node
    assert len(list(sample.nodes)) == len(list(sample.representations))
    return [str(x.node.type) for x in sample.nodes]


def is_identifier(sample: Sample) -> List[Hashable]:
    return [x.node.type == "identifier" for x in sample.nodes]


def is_punkt(sample: Sample) -> List[Hashable]:
    puncs = set(string.punctuation)  # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    return [
        all(char in puncs or char in {"▁"} for char in x.node.type)
        for x in sample.nodes
    ]


def is_defected(sample: Sample) -> List[Hashable]:
    bpe_aug_labels = sample.bpe_aug_labels
    return bpe_aug_labels


def token_name(sample: Sample) -> List[Hashable]:
    bpe_aug_labels = sample.bpe_aug_labels
    return bpe_aug_labels


PROBINGS: List[TokensClassificationProbingTask] = []
PROBINGS.extend(
    [
        TokensClassificationProbingTask(
            "Token Path Type",
            get_path_class,
            description="""
            Classification of types of paths from the root to the node.
        """,
        ),
        # TokensClassificationProbingTask(
        #     "Variable Name",
        #     token_name,
        #     skip_label=0,
        #     code_aug_type="identname",
        #     description="""
        #     Variable Name Prediction (for most popular variable names)
        # """,
        # ),
    ]
)
PROBINGS.extend(
    [
        # TokensClassificationProbingTask(
        #     "Token Node Type",
        #     get_node_type,
        #     description="""
        #     Classification of the type of AST node
        # """,
        # ),
        # TokensClassificationProbingTask(
        #     "Token Is Identifier",
        #     is_identifier,
        #     description="""
        #     Binary Classification: is identifier
        # """,
        # ),
        # TokensClassificationProbingTask(
        #     "Token Is Punkt",
        #     is_punkt,
        #     description="""
        #     Binary Classification: is punktuation
        # """,
        # ),
        # TokensClassificationProbingTask(
        #     "Bracket Is Defected",
        #     is_defected,
        #     skip_label=0,
        #     code_aug_type="brackets",
        #     description="""
        #     Binary Classification: is bracket {}[]()<> defected
        #     `0` label is not used
        # """,
        # ),
        # TokensClassificationProbingTask(
        #     "Variable Is Undeclared",
        #     is_defected,
        #     skip_label=0,
        #     code_aug_type="unidentified_var",
        #     description="""
        #     Binary Classification: is variable inside print statement not declarated
        #     `0` label is not used
        # """,
        # ),
    ]
)


class TokensClassificationProbingTaskVarMis(TokensClassificationProbingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(
            train_data, self.max_classes
        )  # preprocess targets to enumerate them

        def do(data: List[Sample], layers: List[int]) -> Dataset:
            X_by_layer = defaultdict(list)
            y_list: List[Union[float, int]] = []
            samples: List[str] = []
            node_types: List[str] = []
            token2id = []
            all_bpes = []
            all_node_types = []
            sent_ids = []
            for id_, elem in enumerate(data):
                features = elem.features(handle=self.embedding_handle)
                targets = self._get_target(elem)
                targets = self._map_target(targets)

                ids_in = [
                    (idx_, val)
                    for idx_, val in enumerate(targets)
                    if val < self.max_classes
                ]
                if len(ids_in) > 0:

                    for layer in layers:
                        assert len(features) > layer
                        features_layer = features[layer].numpy()
                        features_layer = np.concatenate(
                            [features_layer[it][None, :] for it, val in ids_in], axis=0
                        )
                        features_layer = np.mean(features_layer, axis=0)

                        X_by_layer[layer].append(features_layer)

                    sent_ids.append(elem.sentence_id)
                    y_list.append(ids_in[0][1])

                    samples.append(
                        " ".join([elem.bpe.split(" ")[idx[0]] for idx in ids_in])
                    )

                    all_bpes.append(elem.bpe.split(" "))

            y_numpy = np.array(y_list)
            logging.info(f"unique y_numpy: {str(np.unique(y_numpy))}")

            where = y_numpy < self.max_classes  # filter OOV
            y_numpy = y_numpy[where]
            logging.info(f"remained y_numpy: {str(np.unique(y_numpy))}")
            sleep(1)

            samples_numpy = np.array(samples)[where]
            sent_ids = np.array(sent_ids)[where]
            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer][where]
            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return Dataset(
                X_by_layer_numpy,
                y_numpy,
                samples=samples_numpy,
                all_bpes=all_bpes,
                sent_ids=sent_ids,
            )

        return do(train_data, layers), do(test_data, layers)


PROBINGS.append(
    TokensClassificationProbingTask(
        "Variable Is Undeclared Hard Mean",
        is_defected,
        skip_label=[0, 3],
        code_aug_type="undeclared",
        description="""
            Binary Classification: is variable inside print statement not declarated
            `0` label is not used
        """,
    ),
)


class VarNameClassTask(TokensClassificationProbingTask):
    """for variable naming we need to get a mean embedding for all occurancies of a variable"""

    def __init__(
        self,
        name,
        _get_target: Callable[
            [Sample], List[Tuple[int, str]]
        ],  # idx of variable, and orig variable name
        code_aug_type="identity",
        description="",
        max_classes=15,
        skip_label: Optional[Any] = None,
        var_ident="var",
    ):
        def get_target(sample: Sample):  # to match TokensClassificationProbingTask
            targets: List[Tuple[int, str]] = _get_target(sample)
            if len(targets) == 0:
                return []
            else:
                assert [
                    targets[0][1] == targets[i][1] for i in range(len(targets))
                ], targets  # all occurencies of var name should match
                return [targets[0][1]]  # true variable name

        super().__init__(
            name, get_target, code_aug_type, description, max_classes, skip_label
        )
        # self.var_ident = var_ident  # label of variable (currently not used)
        self._get_target_all_occur = _get_target

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

    def _make_dataset(
        self,
        train_data: List[Sample],
        test_data: List[Sample],
        layers: List[int],
        **kwargs,
    ) -> Tuple[Dataset, Dataset]:
        logging.info("make dataset")

        self.r2id = self.make_dict(train_data, self.max_classes)

        def do(data, layers):
            X_by_layer = defaultdict(list)
            y_list = []
            samples = []
            for data_sample in data:
                features = data_sample.features(handle="none")
                targets: List[Tuple[int, str]] = self._get_target_all_occur(data_sample)
                if len(targets) == 0:
                    # no target here
                    continue
                for layer in layers:
                    assert len(features) > layer
                    gathered = torch.cat(
                        [features[layer][i].unsqueeze(0) for i, _ in targets], dim=0
                    )
                    mean_hidden = gathered.mean(0)
                    X_by_layer[layer].append(mean_hidden.numpy())
                y_list.extend(self._map_target(self.get_target(data_sample)))
                samples.append(data_sample)

            y_numpy = np.array(y_list)
            where = y_numpy < self.max_classes  # filter OOV
            y_numpy = y_numpy[where]
            samples_numpy = np.array(samples)[where]
            logging.info(f"remained y_numpy: {str(np.unique(y_numpy))}")
            sleep(2)

            X_by_layer_numpy = {}
            for layer in layers:
                X_by_layer_numpy[layer] = np.array(X_by_layer[layer])
                X_by_layer_numpy[layer] = X_by_layer_numpy[layer][where]

            logging.info(f"class distribution: {Counter(y_numpy).most_common()}")
            return X_by_layer_numpy, y_numpy, samples_numpy

        X_train_by_layer, y_train, samples_train = do(train_data, layers)
        X_test_by_layer, y_test, samples_test = do(test_data, layers)
        assert len(X_train_by_layer[0]) == len(y_train), (
            len(X_train_by_layer[0]),
            len(y_train),
        )
        return Dataset(X_train_by_layer, y_train, samples_train), Dataset(
            X_test_by_layer, y_test, samples_test
        )
        #     X_test_by_layer, y_test, ["" for _ in range(len(y_test))]
        # )
        # return Dataset(X_train_by_layer, y_train, ["" for _ in range(len(y_train))]), Dataset(
        #     X_test_by_layer, y_test, ["" for _ in range(len(y_test))]
        # )   # for this task simple bound model does not take any input (most common class predictor)


def token_name(sample: Sample) -> List[Tuple[int, str]]:
    # returns list of idx, true_var_name
    variable_names = sample.bpe.split(" ")
    bpe_aug_labels = sample.bpe_aug_labels
    assert len(variable_names) == len(bpe_aug_labels)
    targets = []
    for i, (var_name, label) in enumerate(zip(variable_names, bpe_aug_labels)):
        if label == 0:
            pass  # no label
        else:
            assert var_name == "▁var", (var_name, label)
            targets.append((i, label))
    return targets


PROBINGS.append(
    VarNameClassTask(
        "Variable Name All Occ",
        token_name,
        skip_label=0,
        code_aug_type="identname",
        description="""
        Variable Name Prediction (for most popular variable names)
    """,
    ),
)

if __name__ == "__main__":
    sample = Sample.default()
    for task in PROBINGS:
        print("Task:", task.get_name())
        print("Description:", task.get_description())
        print("Sample:", sample.bpe)
        print("Targets:", task._get_target(sample))
