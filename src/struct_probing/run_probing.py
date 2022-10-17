import argparse
import logging
import logging.handlers
import os
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils import Setup, load_data
# import pandas as pd
from src.struct_probing.probings import (ProbingModelType, ProbingTask,
                      supported_dataset_specific_probings, supported_probings)

from src.struct_probing.utils.sample import Sample


def get_dataset(
    args: argparse.Namespace, code_aug_type: str, dataset_type: str, embedding_type: str
) -> Tuple[List[Sample], Dict[int, List[int]]]:
    # return train, test data

    data, train_ids, test_ids = load_data(
        dataset_type,
        code_aug_type,
        args.model_name,
        embedding_type,
        base_path=args.base_path,
        debug=args.debug,
        mode="all",
        data_dir=args.result_dir,
        post_aug=args.post_aug,
    )
    logging.info("converting dataset")
    # data = data[::100]
    dataset = [Sample(elem, validate=args.validate_sample) for elem in data]

    train_ids = {int(key): value for key, value in train_ids.items()}
    test_ids = {int(key): value for key, value in test_ids.items()}

    return dataset, train_ids, test_ids


def get_savepath(
    args, code_aug_type, probing, probing_model_name, dataset_type, embedding_type
):
    # set up saving path for results
    base_path = Path(
        Setup.get_raw_path(
            dataset_type,
            code_aug_type,
            args.model_name,
            embedding_type,
            args.base_path,
            data_dir=args.result_dir,
            post_aug_name=args.post_aug,
        ),
        "probing_results",
        probing.get_name(),
        probing_model_name,
    )

    if args.debug:
        save_path = Path(
            base_path,
            "debug",
            args.result_file,
        )
    else:
        save_path = Path(
            base_path,
            args.result_file,
        )
    return save_path


def main(args):
    probing: ProbingTask = supported_probings[args.probing]
    # probing defines code augmentation (default: identity)
    warnings.warn(f"setting code_aug_type: {probing.get_augmentation()}")
    # set up code augmentation
    code_aug_type = probing.get_augmentation()
    dataset_type = probing.get_dataset()
    embedding_type = probing.get_embedding_type()
    # set up probing model
    probing_model_name = str(args.probing_model)
    if args.probing_model == ProbingModelType.LOWERBOUND_NGRAM:
        assert args.n_gram_csz is not None
        probing_model_name += f"_{args.n_gram_csz}"

    save_path = get_savepath(
        args, code_aug_type, probing, probing_model_name, dataset_type, embedding_type
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if args.if_no_result and save_path.exists():
        logging.info("already exists, skip")
        return

    dataset, train_ids, test_ids = get_dataset(
        args, code_aug_type, dataset_type, embedding_type
    )

    sample = dataset[0]

    def layers2name(layer):
        return sample.layers[layer]

    layer_range = list(range(len(sample.layers)))
    logging.info(f"LOGGING PATH: {str(save_path.parent)}/log_1.txt")
    handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", f"{str(save_path.parent)}/log_1.txt")
    )
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)

    with open(f"{str(save_path.parent)}/error_1.txt", "w") as error_log:
        try:
            if args.probing_model in [ProbingModelType.LINEAR, ProbingModelType.MLP]:
                result = probing.probe(
                    sample,
                    args.model_name,
                    args.dataset_name,
                    code_aug_type,
                    args.embeddings_name,
                    args.probing_model,
                    dataset,
                    train_ids,
                    test_ids,
                    layer_range=layer_range,
                    layer2name=layers2name,
                    save_dir=str(save_path.parent),
                    # k_folds=args.k_folds,
                )
            elif args.probing_model == ProbingModelType.LOWERBOUND:
                result = probing.probe_lowerbound(
                    sample,
                    args.dataset_name,
                    code_aug_type,
                    args.embeddings_name,
                    dataset,
                    train_ids,
                    test_ids,
                    layer_range=layer_range,
                    layer2name=layers2name,
                    save_dir=str(save_path.parent),
                )
            elif args.probing_model == ProbingModelType.BOW:
                result = probing.probe_bag_of_words(
                    sample,
                    args.dataset_name,
                    code_aug_type,
                    args.embeddings_name,
                    dataset,
                    train_ids,
                    test_ids,
                    layer_range=layer_range,
                    layer2name=layers2name,
                    sample2hashable=probing.get_sample2hashable(),
                )
            # elif args.probing_model == ProbingModelType.LOWERBOUND_NGRAM:
            #     result = probing.probe_lowerbound_ngram(
            #         sample,
            #         args.dataset_name,
            #         code_aug_type,
            #         args.embeddings_name,
            #         dataset, train_ids, test_ids,
            #         layer_range=layer_range,
            #         layer2name=layers2name,
            #         n_gram_csz=args.n_gram_csz,
            #     )
        except Exception as e:
            traceback.print_exc(file=error_log)
            logging.info(str(e))
            logging.info("Error:", traceback.print_exc())
            return

    # result = pd.concat([result, result_lower_bound], ignore_index=True, verify_integrity=True)
    logging.info(str(result))
    if not args.debug:
        result.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default=".", type=str)
    parser.add_argument("--result_dir", default="CodeAnalysis2", type=str)
    parser.add_argument("--dataset_name", default="mlm", type=str)
    # parser.add_argument("--code_aug_type", default="identity", type=str,
    #                     help="changes to input code (e.g. error insertion)")
    parser.add_argument("--model_name", default="MLM", type=str)
    parser.add_argument("--embeddings_name", default="dummy", type=str)
    parser.add_argument(
        "--probing",
        type=str,
        choices=list(supported_probings.keys())
        + list(supported_dataset_specific_probings.keys()),
        default="Token Path Type",
        help="probing task name",
    )
    parser.add_argument(
        "--probing_model",
        type=ProbingModelType,
        choices=list(ProbingModelType),
        default=ProbingModelType.LINEAR,
        help="probing model name",
    )
    parser.add_argument(
        "--post_aug",
        type=str,
        choices=list([aug().name for aug in post_augs]),
        default=None,
        help="ablation augs",
    )
    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--if_no_result", action="store_true")
    parser.add_argument(
        "--n_gram_csz", default=None, type=int, help="context size of ngram model"
    )
    # parser.add_argument("--k_folds", default=3, type=int,
    #                     help="k_folds for train data")
    parser.add_argument("--result_file", default="result_fx.csv", type=str)
    args = parser.parse_args()

    # if args.probing == "Variable Is Undeclared Hard Mean":
    #     print("changed result file")
    #     args.result_file = "result_fx.csv"
    # args.result_file = "result_fx.csv"

    args.validate_sample = args.probing != "Algorithm"

    main(args)
