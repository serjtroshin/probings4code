import json
import logging
import pickle
import random
import traceback
import warnings
from argparse import Namespace
from pathlib import Path
from struct import Struct
from time import sleep
from typing import Any, Dict, Iterable, List

import numpy as np
import sentencepiece as spm
import torch
from src.models.embeddings import Embeddings
from src.models.model import Model
from src.struct_probing.code_augs.aug import (CodeAugmentation,
                                                         LabeledEdge,
                                                         LabeledToken)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SMPEncoder:
    def __init__(self, model_file):
        self.model_file = model_file
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, example):
        code_tokens = self.sp.encode(example, out_type=str)
        return " ".join(code_tokens)


def to_cpu(obj):
    if isinstance(obj, (dict,)):
        for key, value in obj.items():
            obj[key] = to_cpu(value)
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to("cpu")
    elif isinstance(obj, (list, tuple)):
        return [to_cpu(value) for value in obj]
    else:
        return obj


HEADER = Struct("!L")


def send(obj, file):
    """Send a pickled message over the given channel."""
    payload = pickle.dumps(obj, -1)
    file.write(HEADER.pack(len(payload)))
    file.write(payload)


def recv(file):
    """Receive a pickled message over the given channel."""
    header = read_file(file, HEADER.size)
    payload = read_file(file, *HEADER.unpack(header))
    return pickle.loads(payload)


def read_file(file, size):
    """Read a fixed size buffer from the file."""
    parts = []
    while size > 0:
        part = file.read(size)
        if not part:
            raise EOFError
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


class Saver:
    def __init__(self, path, mode=""):
        self.path = Path(path)
        Path(path).mkdir(exist_ok=True, parents=True)
        self.mode = mode
        self.data = []

    def append(self, elem):
        self.data.append(elem)

    def save(self):
        with (self.path / f"data_{self.mode}.pkz").open("wb") as f:
            pickle.dump(self.data, f)
        with (self.path.parent / f"data_{self.mode}_debug.pkz").open("wb") as f:
            pickle.dump(self.data[:10], f)

    def load(self, debug=False):
        path = self.path / f"data_{self.mode}.pkz"
        if debug:
            path = self.path.parent / f"data_{self.mode}_debug.pkz"
        with path.open("rb") as f:
            return pickle.load(f)

    def save_json(self):
        path = self.path / f"{self.mode}.json"
        with path.open("w") as f:
            json.dump(self.data, f)

    def load_json(self):
        path = self.path / f"{self.mode}.json"
        with path.open("r") as f:
            return json.load(f)


# class EmbeddingsSaver:
#     def __init__(self, path, mode=""):
#         self.path = Path(path)
#         Path(path).mkdir(exist_ok=True, parents=True)
#         self.mode = mode
#         self.embeddings = {}  # layer_id -> embeddings

#     def append(self, embeddings: np.ndarray, embeddings_info: List[Any]):
#         assert len(embeddings_info) == len(embeddings)
#         for layer_id, emb in zip(embeddings_info, embeddings):
#             if not layer_id in self.embeddings:
#                 self.embeddings[layer_id] = []
#             self.embeddings[layer_id].append(emb)

#     def save_np(self):
#         for layer_id in self.embeddings:
#             np_data = np.concatenate(self.embeddings[layer_id], axis=0)
#             path = self.path / f"{layer_id}.{self.mode}.npy"
#             with path.open("wb") as f:
#                 np.save(f, np_data, allow_pickle=False)

#     def load_np(self):
#         layer_id2embs = {}
#         for layer_id in self.embeddings:
#             path = self.path / f"{layer_id}.{self.mode}.npy"
#             with path.open("rb") as f:
#                 layer_id2embs[layer_id] = np.load(f)
#         return layer_id2embs


def get_aug_data(
    args: Namespace,
    dataset: Iterable,
    code_aug: CodeAugmentation,
    save_path: Path,
    mode: str,
):
    data = []
    n_skip = 0
    n_total = 0

    dataset = [elem for elem in dataset]

    random.seed(0)
    np.random.seed(0)

    for i, (code, target) in tqdm(
        enumerate(dataset), total=min(len(dataset), args.n_samples)
    ):
        if i == args.n_samples:
            break

        if len(code.split()) > 512:
            code = " ".join(code.split()[:512])

        new_data: List[Dict] = []
        if args.parse_ast:
            tokens, ast = code_aug.process(code)
            for labeled_tokens, labeled_edges, aug_info in code_aug(
                tokens, ast, debug=args.debug
            ):
                code_joined = " ".join([tok.value for tok in labeled_tokens])
                meta = {
                    "code": code,
                    "code_joined": code_joined,  # code after augmentation
                    "target": target,  # not used, if dataset contains some labels
                    "labeled_tokens": [
                        tok.to_json() for tok in labeled_tokens
                    ],  # per token info
                    "labeled_edges": [edge.to_json() for edge in labeled_edges]
                    if labeled_edges is not None
                    else [],  # per edge info
                    "aug_info": aug_info.to_json(),  # per sentence info
                    "sent_id": i,
                }
                new_data.append(meta)
        else:
            code_joined = code
            meta = {
                "code": code,
                "code_joined": code_joined,  # code after augmentation
                "target": target,  # not used, if dataset contains some labels
                "labeled_tokens": None,
                "labeled_edges": None,
                "aug_info": None,
                "sent_id": i,
            }
            new_data.append(meta)

        if len(new_data) == 0:
            warnings.warn("empty output from augmentation")
            n_skip += 1
        n_total += 1

        for elem in new_data:
            data.append(elem)

    print(f"skipped sentences: {n_skip} / {n_total}")
    saver = Saver(save_path, mode=mode)
    saver.data = data
    return saver


def extract_hiddens(
    args: Namespace,
    path: str,
    model: Model,
    dataset: Iterable,
    code_aug: CodeAugmentation,
    embeddings: Embeddings,
    mode,
    n_samples=10000,
    debug=False,
):
    """
    model: Mode, Callable, model(code) -> dict with outputs, required "hidden_states" field
    dataset: Dataset, Iterable, -> X, y
    embeddings: Embeddings, Callable   list of hiddens -> list of output embeddings
    """
    data = Saver(path, mode=mode)
    logging.info(f"save_path: {path}")
    logging.info(f"start of processing {len(data.data)} datapoints")
    sleep(5)

    n_skip = 0
    n_total = 0

    dataset = [_ for _ in dataset]
    invalid_sent_ids = []

    for i, meta in tqdm(enumerate(dataset), total=min(len(dataset), n_samples)):

        if i == n_samples:
            break

        code_joined = meta["code_joined"]

        bpes = model.bpe(code_joined)
        if args.debug:
            print("bpes", bpes)

        if args.parse_ast:
            new_tokens = [
                LabeledToken.from_json(token_json)
                for token_json in meta["labeled_tokens"]
            ]
            labeled_edges = [
                LabeledEdge.from_json(edge_json) for edge_json in meta["labeled_edges"]
            ]

            labeled_bpe = code_aug.bpe_labels(
                new_tokens, bpes
            )  # match each bpe with corresponding label from .augmentation
            if labeled_bpe is None:
                # warnings.warn("bpe match error")
                logging.info("bpe match error")
                # print([token.value for token in new_tokens], bpes)
                # input()
                n_skip += 1
                invalid_sent_ids.append(meta["sent_id"])
                continue
            # input("labeled_edges")
            labeled_edges = code_aug.bpe_edges(labeled_edges, new_tokens, bpes)
            # input("after")
            assert len(labeled_bpe) == len(bpes)
        else:
            labeled_bpe = []
            labeled_edges = []

        if not debug:
            if bpes is None:
                warnings.warn("bpes is None")
                logging.info("bpes is None")
                n_skip += 1
                invalid_sent_ids.append(meta["sent_id"])
                continue
            model_output = model(bpes)
            model_output, labeled_bpe, labeled_edges = code_aug.embeddings_hook(
                model_output, labeled_bpe, labeled_edges
            )
            if model_output is None:
                warnings.warn("model_output is None")
                logging.info("model_output is None")
                n_skip += 1
                invalid_sent_ids.append(meta["sent_id"])
                continue

            output_embeddings = embeddings(model_output)
            # input("output_embeddings")
            if output_embeddings is None:
                warnings.warn("embeddings is None")
                logging.info("embeddings is None")
                n_skip += 1
                invalid_sent_ids.append(meta["sent_id"])
                continue
        else:
            output_embeddings = None

        if "code_joined_no_aug" in meta:
            # input("here")
            bpes = model.bpe(meta["code_joined_no_aug"])
            code_joined_correct = (
                " ".join(bpes).replace(" ", "").replace("▁", " ").strip()
            )
            # print("code_joined_correct", code_joined_correct)
            # input()
            meta["code_joined_correct"] = code_joined_correct
            meta["true_bpe"] = " ".join(bpes)
            # print(meta["true_bpe"])
        else:
            meta["code_joined_correct"] = None
            meta["true_bpe"] = None

        meta = {
            "code": meta["code"],
            "target": meta["target"],
            "outputs": output_embeddings,
            "embeddings_info": model.get_embeddings_info(),
            "embeddings_type": embeddings.type,
            "aug_labels": labeled_bpe,  # per token info
            "labeled_edges": labeled_edges,  # edge labels
            "aug_info": meta["aug_info"],  # per sentence info
            "sent_id": meta["sent_id"],
            "code_joined_correct": meta["code_joined_correct"],
            "true_bpe": meta["true_bpe"],
        }

        n_total += 1

        if debug:
            continue

        try:
            assert len(meta["embeddings_info"]) == len(meta["outputs"]["features"]), (
                len(meta["embeddings_info"]),
                len(meta["outputs"]["features"]),
            )
            # check the number of layers of the models

            if args.parse_ast:
                assert len(meta["aug_labels"]) == len(meta["outputs"]["bpe"].split()), (
                    meta["aug_labels"],
                    meta["outputs"]["bpe"].split(),
                )

            output = meta["outputs"]

            assert len(output["tokens"]) == len(output["bpe"].strip().split()), (
                len(output["tokens"]),
                len(output["bpe"].strip().split()),
            )
            # assert all(
            #     output["features"][i].shape[1] == len(output["tokens"])
            #     for i in range(len(output["features"]))
            # )
        except AssertionError as e:
            traceback.print_exc()
            print("bpes", bpes)
            n_skip += 1
            invalid_sent_ids.append(meta["sent_id"])
            continue
        data.append(meta)
        # logging.info(len(data.data))

    logging.info(f"skipped sentences: {n_skip} / {n_total}")
    logging.info(f"len(data): {len(data.data)}")
    return data, invalid_sent_ids


class Setup:
    def __init__(
        self, dataset, code_aug, model, embeddings, data_dir, post_aug_name=None
    ):
        self.dataset = dataset
        self.code_aug = code_aug
        self.model = model
        self.embeddings = embeddings
        self.post_aug_name = post_aug_name
        self.data_dir = data_dir

    def get_path(self):
        aug_path = str(self.code_aug.type)
        if not self.post_aug_name is None:
            aug_path += f"__{self.post_aug_name}"

        return Setup.get_raw_path(
            self.dataset.type,
            aug_path,
            self.model.type,
            self.embeddings.type,
            data_dir=self.data_dir,
        )

    @staticmethod
    def base_path(base_path=".", directory="CodeAnalysis1"):
        return Path(base_path, directory)

    @staticmethod
    def get_raw_path(
        dataset_type,
        code_aug_type,
        model_type,
        embeddings_type,
        base_path=".",
        data_dir="CodeAnalysis",
        post_aug_name=None,
    ):
        if post_aug_name is not None:
            code_aug_type += f"__{post_aug_name}"
            print("added", post_aug_name, "to path")

        return Path(
            Setup.base_path(base_path, data_dir),
            dataset_type,
            code_aug_type,
            model_type,
            embeddings_type,
        )

    @staticmethod
    def get_aug_path(
        dataset_type, code_aug_type, base_path=".", data_dir="CodeAnalysis"
    ):
        return Path(
            Setup.base_path(base_path, data_dir),
            dataset_type,
            code_aug_type,
        )

    def __str__(self):
        return f"{self.dataset.type}, {str(self.code_aug.type)}, {self.model.type}, {self.embeddings.type}"


def load_data(
    dataset_type,
    code_aug_type,
    model,
    embeddings,
    base_path,
    mode,
    data_dir,
    post_aug,
    debug=False,
):
    path = str(
        Setup.get_raw_path(
            dataset_type,
            code_aug_type,
            model,
            embeddings,
            base_path=base_path,
            data_dir=data_dir,
            post_aug_name=post_aug,
        )
    )
    logging.info(str(path))

    ids_path = Path(path).parent.parent
    with open(ids_path / "train.json", "r") as f:
        train_ids = json.load(f)
    with open(ids_path / "test.json", "r") as f:
        test_ids = json.load(f)

    saver = Saver(path, mode=mode)
    data = saver.load(debug=debug)
    logging.info(f"loaded {len(data)} samples from {saver.path}")

    aug_path = Setup.get_aug_path(dataset_type, code_aug_type)
    with open(f"{base_path}/{aug_path}/all_invalid_ids.json", "r") as f:
        invalid_ids = set(json.load(f))

    n_total = 0
    n_skipped = 0
    filtered_data = []
    for elem in data:
        if elem["sent_id"] in invalid_ids:
            n_skipped += 1  # some models failed of the sentence
        else:
            filtered_data.append(elem)
        n_total += 1
    logging.info(f"n_skipped/n_total = {n_skipped}/{n_total}")

    return filtered_data, train_ids, test_ids


def process_model(dataset, args, setup, n_samples=10000, debug=False):
    train_data, invalid_train_ids = extract_hiddens(
        args,
        path=setup.get_path(),
        model=setup.model,
        dataset=dataset,
        code_aug=setup.code_aug,
        embeddings=setup.embeddings,
        n_samples=n_samples,
        debug=debug,
        mode="all",
    )
    return (train_data, invalid_train_ids)
