import sentencepiece as spm
import torch
from src.models.utils import to_cpu


class SMPEncoder:
    def __init__(self, model_file):
        self.model_file = model_file
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, example):
        code_tokens = self.sp.encode(example, out_type=str)
        return " ".join(code_tokens)


# encoder = SMPEncoder()
# encoder.encode("def main(): pass")
