from pathlib import Path
from typing import List

import torch
from src.models.model import Model, ModelOutput
from src.models.utils import to_cpu
from transformers import AutoModel, AutoTokenizer


def repl_special_char(x):
    if x.startswith("##"):
        return x.replace("##", "")
    else:
        return f"▁{x}"


class BERT(Model):
    def __init__(self, args=[], type="BERT"):
        super().__init__(type)
        self.args = args
        path = Path(__file__).parent.absolute().resolve()
        model_name = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/{model_name}")
        model = AutoModel.from_pretrained(f"{path}/{model_name}")
        self.model = model
        self.tokenizer = tokenizer
        print(f"loaded {model_name} model and tokenizer")

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @staticmethod
    def get_model(model_type, **kwargs):
        return BERT(type=model_type)

    def bpe(self, code: str, max_positions=512) -> List[str]:
        inp = code
        inp = inp.replace("#", "_")
        tokens = self.tokenizer.tokenize(inp)

        if len(tokens) > max_positions - 2:
            tokens = tokens[: max_positions - 2]

        tokens = list(map(lambda x: repl_special_char(x), tokens))
        # tokens[
        #     0
        # ] = f"▁{tokens[0]}"  # first subtoken was not prefixed with special BPE symbol
        return tokens

    def __call__(self, bpe: List[str]):
        """
        Returns:
            dict
        """
        code = "".join(bpe).replace("▁", " ").strip()

        tok_output = self.tokenizer(code, return_tensors="pt")
        tokens_ids = tok_output["input_ids"]

        max_idx = 512
        if tokens_ids.shape[-1] > max_idx:
            # if input is too long, crop it
            tokens_ids = torch.cat(
                (
                    tokens_ids[..., :1],
                    tokens_ids[..., 1 : max_idx - 1],
                    tokens_ids[..., -1:],
                ),
                dim=-1,
            )

        if torch.cuda.is_available():
            tokens_ids = tokens_ids.cuda()

        with torch.no_grad():
            hidden_states = self.model(input_ids=tokens_ids, output_hidden_states=True)[
                "hidden_states"
            ]

            bpes = list(
                map(
                    lambda x: repl_special_char(x),
                    self.tokenizer.convert_ids_to_tokens(
                        tokens_ids[0], skip_special_tokens=False
                    ),
                )
            )[1:-1]
            # bpes[
            #     0
            # ] = f"▁{bpes[0]}"  # in hugginface first subtoken was not prefixed with special BPE symbol

            features = to_cpu(list(list(map(lambda x: x[:, 1:-1, :], hidden_states))))

            tokens = tokens_ids[0].cpu().clone().numpy()
            tokens = tokens[1:-1]
            # print(len(tokens), len(bpe))
            # print(bpes)
            # print([self.tokenizer.convert_ids_to_tokens(
            #             [i], skip_special_tokens=False
            #         ) for i in tokens])
            # input()

            return ModelOutput(bpes, tokens, features)

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(13)] + []  # no decoder values
