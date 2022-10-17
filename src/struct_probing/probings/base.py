import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Hashable, List, Tuple, Union

import numpy as np
import pandas as pd
from src.struct_probing.probings.models import ProbingModelType
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.struct_probing.utils.sample import Sample

# from torch import typename
# from torch._C import Value


MetricName = str
Metrics = dict[MetricName, float]

import logging

log = logging.Logger("base_probing")


class ProbingDataset(ABC):
    @property
    @abstractmethod
    def X_by_layer(self) -> Dict[int, Any]:
        """input features by layer

        Returns:
            Dict[int, Any]: input as layer2feature
        """
        pass

    @property
    @abstractmethod
    def y(self) -> Any:
        """target for a probing task

        Returns:
            Any: target
        """
        pass

    @property
    @abstractmethod
    def samples(self) -> List[Any]:
        pass

    def __len__(self) -> int:
        return len(self.y)


class ProbingNgramDataset(ABC):
    @property
    @abstractmethod
    def bpes(self) -> List[List[str]]:
        """bpe tokens

        Returns:
            List[List[str]: bpe tokens for each sentence
        """
        pass

    def context_str(self, n) -> List[str]:
        """nramm representation

        Returns:
            List[str]: context representations of tokens
            n[int]: context size
        """
        special = "PAD"
        inputs = []
        for sent_id, tok_id in self.positions:
            context = []
            for i in range(0, n // 2):
                if tok_id - i - 1 >= 0:
                    context.append(self.bpes[sent_id][tok_id - i - 1])
                else:
                    context.append(special)
            context = context[::-1]
            context.append(self.bpes[sent_id][tok_id])
            for i in range(0, n // 2):
                if tok_id + i + 1 < len(self.bpes[sent_id]):
                    context.append(self.bpes[sent_id][tok_id + i + 1])
                else:
                    context.append(special)
            context_str = " ".join(context)
            inputs.append(context_str)
            print(f"bpe: {self.bpes[sent_id][tok_id]} -> context {context_str}")
        assert len(inputs) == len(self.y)
        return inputs

    @property
    @abstractmethod
    def positions(self) -> List[Tuple[int, int]]:
        """[summary]

        Returns:
            List[Tuple[int, int]]: positions of bpes tokens (sent_id, token_id)
        """
        pass

    @property
    @abstractmethod
    def y(self) -> Any:
        """target for a probing task

        Returns:
            Any: target
        """
        pass

    def __len__(self) -> int:
        return len(self.y)


class ProbingModel(ABC):
    @abstractmethod
    def train_linear_model(self, X_train: Any, y_train: Any) -> Any:
        """trains a linear probing model over embeddings data

        Args:
            X_train (Any): train data for a particular layer
            y_train (Any): test data for a particular layer

        Returns:
            Any: the trained model
        """
        pass

    @abstractmethod
    def predict_linear_model(self, model: Any, X: Any) -> Any:
        """predicts for a linear probing model over test data

        Args:
            model (Any): output of self.train_linear_model
            X (Any): data

        Returns:
            Any: prediction (y)
        """
        pass

    @abstractmethod
    def train_upper_bound_model(self, X_train: Any, y_train: Any) -> Any:
        """trains a non-linear model over embeddings data

        Args:
            X_train (Any): train data for a particular layer
            y_train (Any): test data for a particular layer

        Returns:
            Any: the trained model
        """
        pass

    @abstractmethod
    def predict_upper_bound_model(self, model: Any, X: Any) -> Any:
        """predicts for an upper bound (non-linear model)

        Args:
            model (Any): output of self.train_linear_model
            X (Any): data

        Returns:
            Any: prediction (y)
        """
        pass

    @abstractmethod
    def eval_prediction(self, y_true: Any, y_pred: Any) -> Dict[str, Union[float, int]]:
        """get metrics

        Args:
            y_true (Any): true labels
            y_pred (Any): predicted

        Returns:
            dict: named metrics
        """
        pass

    @abstractmethod
    def train_lower_bound(
        self, samples: List[Sample], y_train: List[Any]
    ) -> Callable[[Sample], Any]:
        """get optimal constant prediction

        Returns:
            Callable[[Sample], Any]: a model of optimal target
        """
        pass

    def train_n_gramm_lower_bound(
        self, data: ProbingNgramDataset, n=3
    ) -> Callable[[ProbingNgramDataset], List[Any]]:
        """ngramm baseline model, that accounts for context

        Args:
            n (int, optional): context size. Defaults to 3.

        Returns:
            Callable[[ProbingNgramDataset], Any]: model
        """
        raise NotImplemented

    def predict_lower_bound(
        self, model: Callable[[Sample], Any], samples: List[Sample]
    ) -> List[Any]:
        # forms array of optimal targets
        return [model(sample) for sample in samples]

    def train_bag_of_words(
        self,
        samples: List[Sample],
        y_train: List[Any],
        sample2hashes: Callable[[Sample], List[Hashable]],
    ) -> Any:
        """bag of words model (TF-IDF)

        Args:
            samples (List[Sample]): data
            y_train: List[Any], labels

        Returns:
            Any: model
        """
        raise NotImplemented

    def predict_bag_of_words(
        self,
        model: Any,
        samples: List[Sample],
        sample2hashes: Callable[[Sample], List[Hashable]],
    ) -> Any:
        raise NotImplemented

    def save_model(self, model: Any, save_dir: str):
        Path(save_dir).mkdir(exist_ok=True)
        with open(save_dir + "/model.pkz", "wb") as f:
            pickle.dump(model, f)

    def load_model(self, save_dir: str) -> Any:
        with open(save_dir + "/model.pkz", "rb") as f:
            return pickle.load(f)


class Result:
    def __init__(self, name: str, metrics: Metrics, y_true: List, y_pred: List):
        self.name = name
        self.metrics = metrics
        self.y_true = y_true
        self.y_pred = y_pred

    def __str__(self) -> str:
        def to_str(metrics: Metrics):
            return "\t".join(f"{key}: {value:.3f}" for key, value in metrics.items())

        return f"{self.name}:\t{to_str(self.metrics)}"


class ProbingTask(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_target(self, sample: Sample) -> Any:
        # a function to extract target from Sample
        pass

    @abstractmethod
    def get_probing_model(self) -> ProbingModel:
        pass

    def get_augmentation(self) -> str:
        return "identity"  # can be changed

    def get_dataset(self) -> str:
        return "mlm"

    def get_embedding_type(self) -> str:
        return "dummy"

    def get_sample2hashable(self) -> Callable[[Sample], List[Hashable]]:
        raise NotImplementedError

    @abstractmethod
    def _make_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingDataset, ProbingDataset]:
        """converts data samples into sklearn train/test datasets

        Args:
            train/test (List[Sample]): dataset
            layers (List[int]): layers of neural network

        Returns:
            Tuple[ProbingDataset, ProbingDataset]: train, test dataset
        """
        pass

    def _probe(
        self,
        train_data: ProbingDataset,
        test_data: ProbingDataset,
        probing_model_type: ProbingModelType,
        # k_folds=3,
        save_dir=None,
        max_samples=10000,
    ) -> Dict[int, Tuple[Result, Result]]:
        """apply probing model over the dataset

        Args:
            train_data, test_data: a dataset from `make_dataset`,
            probing_model_type: ProbingModel
            save_dir: Optional[str] where to save probing models
        Returns:
            Dict[int, Tuple[Result, Result]]: train, test results for each layer
        """

        probing_model = self.get_probing_model()
        result_by_layer: Dict[int, Tuple[Result, Result]] = {}
        for layer in train_data.X_by_layer.keys():
            logging.info(f"start for layer {str(layer)}")
            X_train_all = np.array(train_data.X_by_layer[layer])
            y_train_all = np.array(train_data.y)

            if len(X_train_all) > max_samples:
                idx = np.random.choice(
                    np.arange(len(X_train_all)), size=max_samples, replace=False
                )
                X_train_all = X_train_all[idx]
                y_train_all = y_train_all[idx]

            X_test = test_data.X_by_layer[layer]

            result_by_layer[layer] = []
            X_train = X_train_all
            y_train = y_train_all
            logging.info(f"len(X_train): {len(X_train)}")
            logging.info(f"len(X_test): {len(X_test)}")
            logging.info(f"training {str(probing_model_type)}")

            if probing_model_type == ProbingModelType.LINEAR:
                model = probing_model.train_linear_model(X_train, y_train)
                train_pred = probing_model.predict_linear_model(model, X_train)
                test_pred = probing_model.predict_linear_model(model, X_test)
            elif probing_model_type == ProbingModelType.MLP:
                model = probing_model.train_upper_bound_model(X_train, y_train)
                train_pred = probing_model.predict_upper_bound_model(model, X_train)
                test_pred = probing_model.predict_upper_bound_model(model, X_test)
            else:
                raise ValueError(f"probing_model_type: {probing_model_type}")

            train = probing_model.eval_prediction(y_train, train_pred)
            test = probing_model.eval_prediction(test_data.y, test_pred)
            if hasattr(test_data, "y_raw"):  # interprete errors
                preds = pd.DataFrame(
                    {
                        "y_true": test_data.y,
                        "y_true_raw": test_data.y_raw,
                        "y_pred": test_pred,
                    }
                )
                print(f"save predictions to {save_dir}")
                preds.to_csv(f"{save_dir}/predictions.csv")

            result_by_layer[layer] = (
                Result("train", train, y_train, train_pred),
                Result("test", test, test_data.y, test_pred),
            )
        return result_by_layer

    # def _lower_bound_ngram(
    #     self, train_dataset: ProbingNgramDataset, test_dataset: ProbingNgramDataset, save_dir=None, n=3
    # ) -> Tuple[Result, Result]:
    #     """apply lower bound probing model

    #     Args:
    #         n[int]: context size

    #     Returns:
    #         Tuple[Result, Result]: train/test metrics
    #     """
    #     probing_model = self.get_probing_model()

    #     model = probing_model.train_n_gramm_lower_bound(train_dataset, n=n)

    #     train_pred = model(train_dataset)
    #     test_pred = model(test_dataset)

    #     train = probing_model.eval_prediction(train_dataset.y, train_pred)
    #     test = probing_model.eval_prediction(test_dataset.y, test_pred)

    #     result = (
    #         Result("train", train, train_dataset.y, train_pred),
    #         Result("test", test, test_dataset.y, test_pred),
    #     )
    #     return result

    def _make_ngram_dataset(
        self, train_data: List[Sample], test_data: List[Sample], layers: List[int]
    ) -> Tuple[ProbingNgramDataset, ProbingNgramDataset]:
        raise NotImplemented

    # def probe_lowerbound_ngram(
    #     self,
    #     sample,
    #     dataset_name: str,
    #     code_aug_type: str,
    #     embeddings_name: str,
    #     train_dataset: List[Sample],
    #     test_dataset: List[Sample],
    #     layer_range: List[int],
    #     layer2name,
    #     save_dir=None,
    #     n_gram_csz=3
    # ):

    #     # sample = Sample.default()
    #     result = pd.DataFrame()

    #     train_data, test_data = self._make_ngram_dataset(train_dataset, test_dataset, layers=layer_range)
    #     train_lower, test_lower = self._lower_bound_ngram(
    #         train_data, test_data, save_dir=save_dir, n=n_gram_csz
    #     )
    #     for layer in layer_range:
    #         for res in train_lower, test_lower:
    #             for metric in res.metrics:
    #                 result = result.append(
    #                     {
    #                         "metric": metric,
    #                         "value": res.metrics[metric],
    #                         "layer": layer2name(layer),
    #                         "task": self.get_name(),
    #                         "description": self.get_description(),
    #                         "code": str(sample.code),
    #                         "bpe": str(sample.bpe),
    #                         "targets": str(self.get_target(sample)),
    #                         "mode": res.name,
    #                         "probing_model_type": ProbingModelType.LOWERBOUND_NGRAM,
    #                         "n_gram_csz": n_gram_csz,
    #                         "dataset_name": dataset_name,
    #                         "code_aug_type": code_aug_type,
    #                         "model_name": "Simple Ngram Bound",
    #                         "embeddings_name": embeddings_name,
    #                     },
    #                     ignore_index=True,
    #                 )

    #     return result

    def _lower_bound(
        self, train_dataset: ProbingDataset, test_dataset: ProbingDataset, save_dir=None
    ) -> Tuple[Result, Result]:
        """apply lower bound probing model

        Returns:
            Tuple[Result, Result]: train/test metrics
        """
        probing_model = self.get_probing_model()

        model = probing_model.train_lower_bound(train_dataset.samples, train_dataset.y)
        # if save_dir is not None:
        #     logging.info(f"saving lower bound at {save_dir}")
        #     probing_model.save_model(model, save_dir)

        train_pred = probing_model.predict_lower_bound(model, train_dataset.samples)
        test_pred = probing_model.predict_lower_bound(model, test_dataset.samples)
        # print(train_pred[0], train_dataset.y[0])

        train = probing_model.eval_prediction(train_dataset.y, train_pred)
        test = probing_model.eval_prediction(test_dataset.y, test_pred)

        # logging.info(str(test_dataset.samples))
        # print("y_true", test_dataset.y)
        # print("y_pred", test)
        # print(f"save predictions to {save_dir}")
        # preds.to_csv(f"{save_dir}/predictions.csv")

        result = (
            Result("train", train, train_dataset.y, train_pred),
            Result("test", test, test_dataset.y, test_pred),
        )
        return result

    def _bag_of_words(
        self,
        train_dataset: ProbingDataset,
        test_dataset: ProbingDataset,
        save_dir=None,
        sample2hashable=None,
    ) -> Tuple[Result, Result]:
        # requires probing_model.sample2hashes(sample) -> List[Hashable]
        probing_model = self.get_probing_model()
        print(probing_model)
        model = probing_model.train_bag_of_words(
            train_dataset.samples, train_dataset.y, sample2hashable
        )

        train_pred = probing_model.predict_bag_of_words(
            model, train_dataset.samples, sample2hashable
        )
        test_pred = probing_model.predict_bag_of_words(
            model, test_dataset.samples, sample2hashable
        )
        print(train_pred[0], train_dataset.y[0])

        train = probing_model.eval_prediction(train_dataset.y, train_pred)
        test = probing_model.eval_prediction(test_dataset.y, test_pred)

        result = (
            Result("train", train, train_dataset.y, train_pred),
            Result("test", test, test_dataset.y, test_pred),
        )
        return result

    def probe_lowerbound(
        self,
        sample,
        dataset_name: str,
        code_aug_type: str,
        embeddings_name: str,
        dataset: List[Sample],
        train_ids: Dict[int, List[int]],
        test_ids: Dict[int, List[int]],
        layer_range: List[int],
        layer2name,
        save_dir=None,
    ):

        # sample = Sample.default()
        result = pd.DataFrame()
        for fold_id in train_ids.keys():
            logging.info(f"{fold_id + 1}/{len(train_ids)} fold:")
            train_set = set(train_ids[fold_id])
            test_set = set(test_ids[fold_id])
            train_dataset = [elem for elem in dataset if elem.sentence_id in train_set]
            test_dataset = [elem for elem in dataset if elem.sentence_id in test_set]
            random.seed(0)
            random.shuffle(train_dataset)
            train_data, test_data = self._make_dataset(
                train_dataset, test_dataset, layers=layer_range
            )
            train_lower, test_lower = self._lower_bound(
                train_data, test_data, save_dir=save_dir
            )
            for layer in layer_range:
                for res in train_lower, test_lower:
                    for metric in res.metrics:
                        result = result.append(
                            {
                                "metric": metric,
                                "value": res.metrics[metric],
                                "layer": layer2name(layer),
                                "task": self.get_name(),
                                "description": self.get_description(),
                                "code": str(sample.code),
                                "bpe": str(sample.bpe),
                                "targets": str(self.get_target(sample)),
                                "mode": res.name,
                                "probing_model_type": ProbingModelType.LOWERBOUND,
                                "dataset_name": dataset_name,
                                "code_aug_type": code_aug_type,
                                "model_name": "Simple Bound",
                                "embeddings_name": embeddings_name,
                                "fold_id": fold_id,
                            },
                            ignore_index=True,
                        )

        return result

    def probe(
        self,
        sample,
        model_name: str,
        dataset_name: str,
        code_aug_type: str,
        embeddings_name: str,
        probing_model: ProbingModelType,
        dataset: List[Sample],
        train_ids: Dict[int, List[int]],
        test_ids: Dict[int, List[int]],
        layer_range: List[int],
        layer2name,
        save_dir=None,
        # k_folds=3,
    ) -> pd.DataFrame:

        # sample = Sample.default()
        result = pd.DataFrame()

        for fold_id in tqdm(train_ids.keys()):
            logging.info(f"{fold_id + 1}/{len(train_ids)} fold:")
            train_set = set(train_ids[fold_id])
            test_set = set(test_ids[fold_id])
            train_dataset = [elem for elem in dataset if elem.sentence_id in train_set]
            test_dataset = [elem for elem in dataset if elem.sentence_id in test_set]
            random.seed(0)
            random.shuffle(train_dataset)
            train_data, test_data = self._make_dataset(
                train_dataset, test_dataset, layers=layer_range
            )
            result_by_layer = self._probe(
                train_data, test_data, probing_model, save_dir=save_dir
            )
            for layer, (train, test) in tqdm(result_by_layer.items()):
                for res in train, test:
                    assert isinstance(res, Result)
                    for metric in res.metrics:
                        result = result.append(
                            {
                                "metric": metric,
                                "value": res.metrics[metric],
                                "layer": layer2name(layer),
                                "task": self.get_name(),
                                "description": self.get_description(),
                                "code": str(sample.code),
                                "bpe": str(sample.bpe),
                                "targets": str(self.get_target(sample)),
                                "mode": res.name,
                                "probing_model_type": probing_model,
                                "dataset_name": dataset_name,
                                "code_aug_type": code_aug_type,
                                "model_name": model_name,
                                "embeddings_name": embeddings_name,
                                "fold_id": fold_id,
                            },
                            ignore_index=True,
                        )

        return result

    def probe_bag_of_words(
        self,
        sample,
        dataset_name: str,
        code_aug_type: str,
        embeddings_name: str,
        dataset: List[Sample],
        train_ids: Dict[int, List[int]],
        test_ids: Dict[int, List[int]],
        layer_range: List[int],
        layer2name,
        save_dir=None,
        sample2hashable=None,
    ):

        # sample = Sample.default()
        result = pd.DataFrame()
        for fold_id in train_ids.keys():
            logging.info(f"{fold_id + 1}/{len(train_ids)} fold:")
            train_set = set(train_ids[fold_id])
            test_set = set(test_ids[fold_id])
            train_dataset = [elem for elem in dataset if elem.sentence_id in train_set]
            test_dataset = [elem for elem in dataset if elem.sentence_id in test_set]
            random.seed(0)
            random.shuffle(train_dataset)
            train_data, test_data = self._make_dataset(
                train_dataset, test_dataset, layers=layer_range
            )
            train_lower, test_lower = self._bag_of_words(
                train_data,
                test_data,
                save_dir=save_dir,
                sample2hashable=sample2hashable,
            )
            for layer in layer_range:
                for res in train_lower, test_lower:
                    for metric in res.metrics:
                        result = result.append(
                            {
                                "metric": metric,
                                "value": res.metrics[metric],
                                "layer": layer2name(layer),
                                "task": self.get_name(),
                                "description": self.get_description(),
                                "code": str(sample.code),
                                "bpe": str(sample.bpe),
                                "targets": str(self.get_target(sample)),
                                "mode": res.name,
                                "probing_model_type": ProbingModelType.BOW,
                                "dataset_name": dataset_name,
                                "code_aug_type": code_aug_type,
                                "model_name": "BOW",
                                "embeddings_name": embeddings_name,
                                "fold_id": fold_id,
                            },
                            ignore_index=True,
                        )

        return result
