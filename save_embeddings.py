import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

from src.models.available_models import FINETUNED_MODELS, MODELS
from src.models.embeddings import Embeddings
from src.models.model import Model
from src.struct_probing.code_augs import available_augs
from src.struct_probing.code_augs.aug import CodeAugmentation
from src.utils import Saver, Setup, process_model
from tasks.mlm.dataset import AugDataset

log = logging.getLogger("save_embeddings")
log.setLevel(logging.INFO)


def iter_pretrained_models(args) -> Model:
    for name, model in MODELS.items():
        if args.model == "all" or name == args.model:
            if not args.preview:
                model = model.get_model(
                    name,
                    debug=args.debug,
                    task_args=args,
                )
            else:
                model = None
            logging.info(f"Processing: {name}, {str(model)}")
            yield model


def iter_finetuned_models(args) -> Model:
    for model_name, (cls, checkpoint_path) in FINETUNED_MODELS.items():
        if args.model == "all" or model_name == args.model:
            if not args.preview:
                model = cls.get_model(
                    model_name, checkpoint_path=checkpoint_path, task_args=args
                )
            else:
                model = None
            logging.info(f"Processing: {model_name}, {str(model)}")
            yield model


def iter_models(args) -> Model:
    for model in iter_pretrained_models(args):
        yield model
    for model in iter_finetuned_models(args):
        yield model


def get_code_augmentation(args) -> CodeAugmentation:
    return available_augs[args.insert]()


def get_embeddings_data(args, setup: Setup, dataset) -> Tuple[Saver, List[int]]:
    return process_model(
        dataset, args, setup, n_samples=args.n_samples, debug=args.debug
    )


def main(args):
    code_aug = get_code_augmentation(args)
    if code_aug.required_dataset() is not None:
        warnings.warn(
            f"changed task data: {args.task} -> {code_aug.required_dataset()}"
        )
        args.task = code_aug.required_dataset()

    save_path = str(Setup.get_aug_path(args.task, args.insert))
    dataset = AugDataset(
        Saver(save_path, mode="all").load_json(),
        type=args.task,
    )
    embeddings = Embeddings(
        code_aug.required_embeddings(), pairsent=False
    )  # if pairsent is True: sent1

    for model in iter_models(args):
        if not args.preview:

            setup = Setup(dataset, code_aug, model, embeddings, data_dir=args.input_dir)

            logging.info(f"Setup: {str(setup)}")

            (saver, invalid_ids) = get_embeddings_data(args, setup, dataset)
            if not args.debug:
                logging.info(f"saver: {len(saver.data), saver.path, saver.mode}")
                saver.save()
                with Path(setup.get_path(), "invalid_ids.json").open("w") as f:
                    json.dump(invalid_ids, f)
            else:
                warnings.warn("Data is not saved in DEBUG mode!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        help="model",
        choices=list(MODELS.keys()) + list(FINETUNED_MODELS.keys()) + ["all"],
    )
    parser.add_argument("--task", default="mlm", help="data-dir")
    parser.add_argument("--input_dir", default="CodeAnalysis", help="data-dir")
    # parser.add_argument("--embeddings", default="dummy", type=str)
    # parser.add_argument("--sbatch", action="store_true", help="run in parallel on slurm cluster")
    parser.add_argument("--n_samples", type=int, default=10000)

    parser.add_argument(
        "--insert",
        type=str,
        choices=list(available_augs.keys()),
        default="identity",
        help="data augmentation for probing tasks (bug detection)",
    )

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    args.parse_ast = args.insert != "sorts" and not args.insert.startswith(
        "algo"
    )
    if args.insert.startswith("algo"):
        args.lang = "python"

    main(args)
