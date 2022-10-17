import argparse
import sys
import warnings
from pathlib import Path
from typing import List

import torch
from src.models.model import Model, ModelOutput
from data.github.preprocessing.src.code_tokenizer import (tokenize_java,
                                                          tokenize_python)
from src.models.plbart.bart_classifier import BartClassifier

CUR_DIR = Path(__file__).parent

def get_classification_task_parser(save_dir=CUR_DIR, plbart=CUR_DIR):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=f"{save_dir}")
    parser.add_argument("--user_dir", default=f"{plbart}/source")
    parser.add_argument("--model_name", default=None)
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

    parser.add_argument("--no_token_positional_embeddings", action="store_true")
    return parser


class PLBARTModel(Model):
    def __init__(self, args, type="MLM", debug=False, task_args=None):
        super().__init__(type)
        self.args = args
        self.task_args = task_args

        self.model = BartClassifier(self.args, debug=debug)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @staticmethod
    def get_model(
        model_type="MLM",
        debug=False,
        checkpoint_path=None,
        task_args=None,
    ):
        """
        creates a model
        """
        if model_type == "plbart_large":
            raise ValueError("you should use PLBARTModelLarge")
        elif model_type.startswith("finetuned"):
            checkpoint = Path(__file__).partent / checkpoint_path / "checkpoint_best.pt"
        else:
            checkpoint = "plbart_base.pt"  # plbart base
            checkpoint = Path(__file__).parent / checkpoint

        assert checkpoint.exists(), checkpoint
        MODEL_DIR = Path(checkpoint).parent
        CHECKPOINT = Path(checkpoint).name
        print(f"starting processing {CHECKPOINT} from {MODEL_DIR}")
        
        args = f"--model_dir {MODEL_DIR} --model_name {CHECKPOINT} "
        args = args.split()

        parser = get_classification_task_parser()

        args = parser.parse_args(args)
        model = PLBARTModel(args, type=model_type, debug=debug, task_args=task_args)
        
        return model

    def bpe(self, code: str) -> List[str]:
        if not self.task_args.parse_ast:
            # print(code)
            if self.task_args.lang == "java":
                tok = tokenize_java
            elif self.task_args.lang == "python":
                tok = tokenize_python
            else:
                raise ValueError()
            code = " ".join(tok(code)).strip()
            if len(code) == 0:
                return None
        return self.model.bpe(code)

    def __call__(self, bpe: List[str]) -> ModelOutput:
        """[summary]

        Args:
            bpe (List[str]): [description]

        Returns:
            List[Any]: features by layers
        """
        return self.model.extract_features(bpe, return_all_hiddens=True)

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(7)] + [f"d{i}" for i in range(7)]


class PLBARTModelLarge(PLBARTModel):
    def __init__(self, args, type="PLBART_large", debug=False, task_args=None):
        super().__init__(args, type, debug=debug, task_args=task_args)

    @staticmethod
    def get_model(
        model_type="MLM",
        debug=False,
        checkpoint_path=None,
        task_args=None,
    ):
        """
        creates a model
        """
        # PLBART_PATH = Path(__file__, "../../../../..").absolute().resolve()
        # print(PLBART_PATH)
        # sys.path.insert(0, str(PLBART_PATH))

        # TASK = "mlm"  # model type
        # task_dir = f"{PLBART_PATH}/interprete/tasks/{TASK}"

        assert model_type == "plbart_large"
        checkpoint = Path("plbart_large.pt")

        assert checkpoint.exists(), checkpoint
        MODEL_DIR = Path(checkpoint).parent
        CHECKPOINT = Path(checkpoint).name
        print(f"starting processing {CHECKPOINT} from {MODEL_DIR}")
        # if not debug:
        args = f"--model_dir {MODEL_DIR} --model_name {CHECKPOINT} "
        # else:
        #     args = f"--arch mbart_small --user-dir ../fairseq_plugins"

        args = args.split()

        parser = get_classification_task_parser()

        args = parser.parse_args(args)
        model = PLBARTModelLarge(
            args, type=model_type, debug=debug, task_args=task_args
        )
        return model

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(13)] + [f"d{i}" for i in range(13)]
