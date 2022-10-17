import argparse
import json
import logging
import shutil
import warnings
from collections import Counter, defaultdict
from cProfile import label
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.available_models import MODELS
from src.struct_probing.code_augs import (CodeAugmentation,
                                                     available_augs, post_augs)
from src.struct_probing.code_augs.aug import (LabeledToken,
                                                         LabeledTokenTyped)
from src.utils import Saver, Setup, get_aug_data
from tasks import DATASETS
from tasks.mlm.dataset import AugDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

log = logging.getLogger("save_embeddings")
log.setLevel(logging.INFO)


def get_dataset(args):
    return DATASETS[args.task]()


def iter_models(args):
    if args.model == "all":
        models = MODELS.items()
    else:
        models = list(filter(lambda x: x[0] == args.model, MODELS.items()))
    for name, model in models:
        yield name, model


def get_code_augmentation(args) -> CodeAugmentation:
    aug = available_augs[args.insert]()
    if aug.type != args.insert:
        warnings.warn(f"aug.type != args.insert, {aug.type}, {args.insert}")
    # aug.type = args.insert
    return aug


def apply_post_sugs(args, saver: Saver) -> Saver:
    """
    Returns:
        Saver: saver with additional augmentation applied for data
    """


def main(args):
    global post_augs
    code_aug = get_code_augmentation(args)
    if code_aug.required_dataset() is not None:
        warnings.warn(
            f"changed task data: {args.task} -> {code_aug.required_dataset()}"
        )
        args.task = code_aug.required_dataset()

    read_path = str(
        Setup.get_aug_path(args.task, args.insert, data_dir=args.data_dir)
    )
    dataset = AugDataset(Saver(read_path, mode="all").load_json(), type=args.task)
    print(dataset[0].keys())

    basic_aug = CodeAugmentation()
    print("post_augs", [aug().name for aug in post_augs])

    dataset_per_aug = defaultdict(list)
    counter = Counter()

    for ii, elem in enumerate(tqdm(dataset)):

        code = elem["code_joined"]
        labeled_tokens = [LabeledToken.from_json(t) for t in elem["labeled_tokens"]]
        code_joined = " ".join(map(lambda x: x.value, labeled_tokens))
        tokens, ast = basic_aug.process(code)
        types = [tok.node.type for tok in tokens]
        if "ERROR" in types:
            continue
        counter.update(types)
        assert len(tokens) == len(labeled_tokens), (len(tokens), len(labeled_tokens))
        for a, b in zip(tokens, labeled_tokens):
            assert a.value == b.value

        labeled_tokens_typed = [
            LabeledTokenTyped(lab_token.value, token.node.type, lab_token.label)
            for lab_token, token in zip(labeled_tokens, tokens)
        ]

        for post_aug in post_augs:
            post_aug = post_aug()
            if ii < 3:
                print("-" * 40, post_aug.name, "-" * 40, flush=True)

            new_labeled_tokens = [post_aug(tok) for tok in labeled_tokens_typed]
            code_joined_new = " ".join(map(lambda x: x.value, new_labeled_tokens))
            if ii < 3:
                print(code_joined_new, flush=True)

            meta = {
                "code": elem["code"],
                "code_joined": code_joined_new,
                "code_joined_no_aug": code_joined,
                "target": elem["target"],
                "labeled_tokens": [t.to_json() for t in new_labeled_tokens],
                "labeled_edges": elem["labeled_edges"],
                "aug_info": elem["aug_info"],
                "sent_id": elem["sent_id"],
            }
            dataset_per_aug[post_aug.name].append(meta)

        if args.debug:
            if ii % 100 == 0:
                pass
                # print(counter.most_common())
        #     input("press any key")

    if not args.debug:
        for name in dataset_per_aug.keys():
            print(name, len(dataset_per_aug[name]))
            # print(dataset_per_aug[name][0])

            save_path = Setup.get_aug_path(
                args.task, f"{args.insert}__{name}", data_dir=args.output_dir
            )
            print(f"saving {name} to {save_path}")
            saver = Saver(save_path, mode="all")
            saver.data = dataset_per_aug[name]
            saver.save_json()

            for file in "train.json", "test.json":
                shutil.copy(Path(read_path, file), Path(save_path, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="CodeAnalysis", help="data-dir")
    parser.add_argument("--output_dir", default="CodeAnalysisAug", help="output-dir")

    parser.add_argument("--task", default="mlm", help="data-dir")
    parser.add_argument("--n_samples", type=int, default=1000000)

    parser.add_argument(
        "-i",
        "--insert",
        type=str,
        choices=list(available_augs.keys()),
        default="identity",
        help="data augmentation for probing tasks (bug detection)",
    )

    # Debug
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    main(args)
