import argparse
import logging
import warnings
from pathlib import Path
from time import sleep

from src.utils import Setup

log = logging.getLogger("run_parallel")
log.setLevel(logging.DEBUG)

from code_augs import post_augs
from src.struct_probing.probings import ProbingModelType, ProbingTask, supported_probings
from supported_models import FINETUNED_MODELS, PRETRAINED_MODELS

from utils.slurm import get_slurm_str, submit_job, submit_local


def iter_models(args):
    # if args.block == "pretrained":
    for model_name in PRETRAINED_MODELS:
        yield model_name
    # elif args.block == "finetuned":
    for model_name in FINETUNED_MODELS:
        yield model_name


def get_probings(args):
    if args.probing == "all":
        return supported_probings
    else:
        return {
            key: probing
            for key, probing in supported_probings.items()
            if key == args.probing
        }


def get_path(
    dataset_name,
    code_aug_type,
    model_name,
    embeddings_name,
    probing,
    probing_mode,
    post_aug,
):
    result_file = args.result_file
    result_file = "result_fx.csv"
    # if probing == "Variable Is Undeclared Hard Mean":
    #     result_file = "result_fx.csv"

    base_path = Path(
        Setup.get_raw_path(
            dataset_name,
            code_aug_type,
            model_name,
            embeddings_name,
            ".", # "../..",
            post_aug_name=post_aug,
        ),
        "probing_results",
        probing,
        str(probing_mode),
    )
    save_path = Path(
        base_path,
        result_file,
    )
    return save_path


def iter_tasks(args):
    probings = get_probings(args)
    for model_name in iter_models(args):
        # continue
        if args.model != "all":
            if model_name != args.model:
                continue
        probing_mode = args.probing_model
        for probing in probings:
            probing_task: ProbingTask = supported_probings[probing]
            # probing defines code augmentation (default: identity)
            # warnings.warn(f"setting code_aug_type: {probing.get_augmentation()}")
            code_aug_type = probing_task.get_augmentation()
            dataset_name = probing_task.get_dataset()
            embeddings_name = probing_task.get_embedding_type()
            save_path = get_path(
                dataset_name,
                code_aug_type,
                model_name,
                embeddings_name,
                probing,
                probing_mode,
                post_aug=args.post_aug,
            )

            if args.if_no_result and save_path.exists():
                pass
                # logging.info(f"already exists, skip : ------  {save_path}")
            else:
                logging.info(save_path)
                data_dir = Path(*save_path.parts[:-4])

                assert (
                    "data_all.pkz" in [p.name for p in data_dir.glob("*")]
                    or model_name == "Codex"
                ), data_dir
                if not "data_all.pkz" in [p.name for p in data_dir.glob("*")]:
                    print("NO DATA", data_dir)
                    continue
                # assert "data_train.pkz" in [p.name for p in data_dir.glob("*")], data_dir

                task_str = f"python3 src/struct_probing/run_probing.py  \
                        --dataset_name '{dataset_name}' \
                        --model_name '{model_name}' \
                        --embeddings_name '{embeddings_name}' \
                        --probing_mode '{probing_mode}' \
                        --probing '{probing}'"
                yield model_name, probing, task_str
                
                if model_name == "MLM":
                    for probing_mode in ProbingModelType.BOW, ProbingModelType.LOWERBOUND:
                        task_str = f"python3 src/struct_probing/run_probing.py  \
                            --dataset_name '{dataset_name}' \
                            --model_name '{model_name}' \
                            --embeddings_name '{embeddings_name}' \
                            --probing_mode '{probing_mode}' \
                            --probing '{probing}'"
                        yield model_name, probing, task_str
    return


def main(args):
    logging.info("start run_parallel.py")
    save_dir = args.save_dir
    Path(save_dir).mkdir(exist_ok=True)
    for model_name, task_name, command_str in iter_tasks(args):
        if args.sbatch:
            slurm_str = get_slurm_str(
                save_dir,
                cpus=args.cpus,
                gpus=args.gpus,
                name=model_name,
                task=task_name,
                constraint=args.constraint,
            )
            command = f'{slurm_str} --wrap "{command_str}"'
            if not args.preview:
                logging.info(command)
                submit_job(command)
                # sleep(2)
        else:
            if not args.preview:
                logging.info(command_str)
                submit_local(command_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="CodeAnalysis/")
    parser.add_argument("--model", default="all", type=str)
    parser.add_argument(
        "--probing", default="all", choices=["all"] + list(supported_probings.keys())
    )
    parser.add_argument(
        "--probing_model",
        type=ProbingModelType,
        choices=list(ProbingModelType),
        default=ProbingModelType.LINEAR,
        help="probing model name",
    )
    parser.add_argument(
        "--sbatch", action="store_true", help="run in parallel on slurm cluster"
    )
    parser.add_argument(
        "--block", default="pretrained", choices=["pretrained", "finetuned"]
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        default="dummy",
        choices=[
            "dummy"
        ],
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
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument(
        "--constraint",
        type=str,
        default="type_b",
        choices=["type_e", "", "type_b", "type_c"],
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--if_no_result", action="store_true")
    parser.add_argument("--result_file", default="result_fx.csv", type=str)

    args = parser.parse_args()

    main(args)
