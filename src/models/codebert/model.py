from pathlib import Path
from typing import List

import torch
from src.models.model import Model, ModelOutput
from src.models.utils import to_cpu
from transformers import AutoModel, AutoTokenizer


class CodeBertModel(Model):
    def __init__(self, args=[], type="CodeBert"):
        super().__init__(type)
        self.args = args
        path = Path(__file__).parent.absolute().resolve()
        print(path)
        tokenizer = AutoTokenizer.from_pretrained(f"{path}/codebert-base")
        model = AutoModel.from_pretrained(f"{path}/codebert-base")
        self.model = model
        self.tokenizer = tokenizer
        print("loaded CodeBert model and tokenizer")

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @staticmethod
    def get_model(model_type, **kwargs):
        return CodeBertModel(type=model_type)

    def bpe(self, code: str, max_positions=512) -> List[str]:
        inp = code
        inp = inp.replace("▁", "_")
        tokens = self.tokenizer.tokenize(inp)
        if len(tokens) > max_positions - 2:
            tokens = tokens[: max_positions - 2]

        tokens = list(map(lambda x: x.replace("Ġ", "▁"), tokens))
        tokens[
            0
        ] = f"▁{tokens[0]}"  # first subtoken was not prefixed with special BPE symbol
        return tokens

    def __call__(self, bpe: List[str]):
        """
        Returns:
            dict
        """
        code = "".join(bpe).replace("▁", " ").strip()
        # "▁" symbol in code resulted in ['Ġ', 'â', 'ĸ', 'ģ'] tokens

        # inp = "hello i am Sergey"
        code_tokens = self.tokenizer.tokenize(code)
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        tokens_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))[None, :]
        # print("token_ids", tokens_ids.shape)

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
        # simply generate one code span

        with torch.no_grad():
            generated_ids = self.model.forward(
                torch.tensor(tokens_ids), output_hidden_states=True
            )

            bpes = list(
                map(
                    lambda x: x.replace("Ġ", "▁"),
                    self.tokenizer.convert_ids_to_tokens(
                        tokens_ids[0], skip_special_tokens=True
                    ),
                )
            )
            bpes[
                0
            ] = f"▁{bpes[0]}"  # in hugginface first subtoken was not prefixed with special BPE symbol

            features = to_cpu(
                list(list(map(lambda x: x[:, 1:-1, :], generated_ids.hidden_states)))
            )

            tokens = tokens_ids[0].cpu().clone().numpy()
            tokens = tokens[1:-1]

            return ModelOutput(bpes, tokens, features)

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(13)] + []  # no decoder values
