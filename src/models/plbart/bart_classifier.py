from typing import List

import torch
import sentencepiece as spm
from fairseq.data import data_utils
from fairseq.models.bart import BARTModel
from src.models.utils import to_cpu
# from models.sent.plbart_sent_class.model import BartClassifier
from src.models.model import ModelOutput

import torch


class SMPEncoder:
    def __init__(self, model_file):
        self.model_file = model_file
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, example):
        code_tokens = self.sp.encode(example, out_type=str)
        return " ".join(code_tokens)


class TokenEncoder:
    def __init__(self, sentencepiece):
        self.smp_encoder = SMPEncoder(sentencepiece)

    def encode(self, code: str):
        code = self.smp_encoder.encode(code.strip())
        return code


class BartClassifier(torch.nn.Module):
    def __init__(self, args, debug=False):
        super(BartClassifier, self).__init__()
        self.args = args
        self.smp_encoder = TokenEncoder(self.args.sentencepiece)

        self.bart = None
        self.label_fn = lambda label: label
        bart = BARTModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_name,
            data_name_or_path=args.data_bin_path,
            user_dir=args.user_dir,
            task="plbart_sentence_prediction",
            no_token_positional_embeddings=args.no_token_positional_embeddings,
            debug=debug,
        )
        print("bart loaded")
        if not debug:
            if torch.cuda.is_available():
                bart = bart.cuda()
            bart.eval()
        self.bart = bart
        self.label_fn = lambda label: bart.task.label_dictionary.string(
            [label + bart.task.label_dictionary.nspecial]
        )

    @staticmethod
    def encode_sentence(model, bpe: List[str], max_positions=512) -> torch.Tensor:
        # https://github.com/pytorch/fairseq/blob/108f7204f6ccddb676e6d52006da219ce96a02dc/fairseq/models/bart/hub_interface.py#L33
        # sentence: bpe encoded
        if len(bpe) > max_positions - 2:
            bpe = bpe[: max_positions - 2]
        bpe = " ".join(["<s>"] + bpe + ["</s>"])
        tokens = model.task.source_dictionary.encode_line(
            bpe, add_if_not_exist=False, append_eos=False
        )
        return tokens.long()

    def bpe(self, code: str, max_positions=512) -> List[str]:
        bpe = self.smp_encoder.encode(code)
        if len(bpe.split(" ")) > max_positions - 2:
            bpe = " ".join(bpe.split(" ")[: max_positions - 2])
        return bpe.split(" ")

    def tokens(self, code=None, bpe=None) -> torch.Tensor:
        if bpe is None:
            bpe = self.bpe(code)
        tokens = BartClassifier.encode_sentence(self.bart, bpe)
        return tokens

    def extract_features(self, bpe: List[str], return_all_hiddens=False) -> ModelOutput:
        """
        Args:
            code (str)

        Returns:
        [list]: [list of feature vectors by layers]
        """
        with torch.no_grad():

            tokens = self.tokens(bpe=bpe)

            if torch.cuda.is_available():
                tokens = tokens.cuda()

            hiddens = to_cpu(
                self.bart.extract_features(
                    tokens, return_all_hiddens=return_all_hiddens
                )
            )
            tokens_cpu = tokens.cpu().clone()[1:-1]  # no special tokens

            return ModelOutput(bpe, tokens_cpu, hiddens)

    def __call__(self, code):
        args, bart = self.args, self.bart

        sent = self.smp_encoder.encode(code)
        tokens = BartClassifier.encode_sentence(bart, sent)

        with torch.no_grad():
            batch_input = data_utils.collate_tokens(
                [tokens], bart.model.encoder.dictionary.pad(), left_pad=False
            )
            prediction = bart.predict(args.classification_head_name, batch_input)
            prediction = prediction.argmax(dim=1).cpu().numpy().tolist()
            prediction = [int(self.label_fn(p)) for p in prediction]
            return prediction[0]
