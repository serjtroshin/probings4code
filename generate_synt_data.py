import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
from src.models.available_models import MODELS
from src.struct_probing.code_augs import (CodeAugmentation,
                                                     available_augs)
from src.utils import Setup, get_aug_data
from tasks import DATASETS
from sklearn.model_selection import KFold

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


def main(args):
    logging.info(f"start {args.insert}")
    code_aug = get_code_augmentation(args)

    if code_aug.required_dataset() is not None:
        warnings.warn(
            f"changed task data: {args.task} -> {code_aug.required_dataset()}"
        )
        args.task = code_aug.required_dataset()

    dataset = get_dataset(args)

    save_path = Setup.get_aug_path(
        dataset.type, args.insert, data_dir=args.data_dir
    )
    logging.info(f"Save to: {str(save_path)}")

    saver = get_aug_data(
        args, list(dataset), code_aug, save_path, mode="all"
    )
    
    sent_ids = []
    for elem in saver.data:
        sent_id = elem["sent_id"]
        sent_ids.append(sent_id)
        
    sent_ids = np.array(sent_ids)
    print(f"got {len(sent_ids)} unique sent_ids")

    # train test splits
    train, test = {}, {}
    for i, (train_ids, test_ids) in enumerate(
        KFold(args.prep_folds, random_state=42, shuffle=True).split(sent_ids)
    ):
        train[i] = sent_ids[train_ids].tolist()
        test[i] = sent_ids[test_ids].tolist()
    with open(f"{save_path}/train.json", "w") as f:
        json.dump(train, f)
    with open(f"{save_path}/test.json", "w") as f:
        json.dump(test, f)

    if not args.debug:
        saver.save_json()
    else:
        warnings.warn("Data is not saved in DEBUG mode!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="CodeAnalysis", help="data-dir")

    parser.add_argument("--prep_folds", default=4, help="number of train/test splits")

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
