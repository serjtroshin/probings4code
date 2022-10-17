import argparse
from pathlib import Path

DATA_DIR = str(Path(__file__).parent)


def get_classification_task_parser(plbart):
    save_dir = DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=f"{save_dir}")
    parser.add_argument("--user_dir", default=f"{plbart}/source")
    parser.add_argument("--model_name", default="checkpoint.pt")
    parser.add_argument(
        "--classification_head_name", default="sentence_classification_head"
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_example", default=-1, type=int)
    parser.add_argument(
        "--data_bin_path",
        default=f"{save_dir}/data-bin",
        help="path with input0/dict.txt",
    )
    parser.add_argument("--input_file", default=f"{save_dir}/test.input0")
    parser.add_argument("--label_file", default=f"{save_dir}/test.label")
    parser.add_argument(
        "--sentencepiece", default=f"{save_dir}/sentencepiece.bpe.model"
    )
    return parser
