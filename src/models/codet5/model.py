from pathlib import Path
from typing import List

import torch
from src.models.model import Model, ModelOutput
from src.models.utils import to_cpu
from transformers import (RobertaTokenizer, T5ForConditionalGeneration,
                          T5Tokenizer)


class CodeT5Model(Model):
    def __init__(self, args=[], type="CodeT5", model_name="codet5-base"):
        super().__init__(type)
        self.args = args
        path = Path(__file__).parent.absolute().resolve()
        print(path, model_name)
        if model_name == "t5-base":
            tokenizer = T5Tokenizer.from_pretrained(f"{path}/tokenizer-{model_name}")
        else:
            tokenizer = RobertaTokenizer.from_pretrained(
                f"{path}/tokenizer-{model_name}"
            )
        model = T5ForConditionalGeneration.from_pretrained(f"{path}/model-{model_name}")
        self.model = model
        self.tokenizer = tokenizer
        print(f"loaded {model_name} model")

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @staticmethod
    def get_model(model_type, **kwargs):
        return CodeT5Model(type=model_type)

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
        # print("len(bpe):", len(tokens))
        return tokens

    def __call__(self, bpe: List[str]):
        """
        Returns:
            dict
        """
        code = "".join(bpe).replace("▁", " ").strip()
        print(code)

        # "▁" symbol in code resulted in ['Ġ', 'â', 'ĸ', 'ģ'] tokens

        # inp = "hello i am Sergey"
        input_ids = self.tokenizer(code, return_tensors="pt").input_ids
        max_idx = 512
        if input_ids.shape[-1] > max_idx:
            # if input is too long, crop it
            input_ids = torch.cat(
                (
                    input_ids[..., :1],
                    input_ids[..., 1 : max_idx - 1],
                    input_ids[..., -1:],
                ),
                dim=-1,
            )

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        # simply generate one code span

        with torch.no_grad():
            generated_ids = self.model.forward(
                input_ids, decoder_input_ids=input_ids, output_hidden_states=True
            )

            bpes = list(
                map(
                    lambda x: x.replace("Ġ", "▁"),
                    self.tokenizer.convert_ids_to_tokens(
                        input_ids[0], skip_special_tokens=True
                    ),
                )
            )
            bpes[
                0
            ] = f"▁{bpes[0]}"  # first subtoken was not prefixed with special BPE symbol
            features = to_cpu(
                list(
                    list(
                        map(
                            lambda x: x[:, 1:-1, :], generated_ids.encoder_hidden_states
                        )
                    )
                    + list(
                        map(lambda x: x[:, 2:, :], generated_ids.decoder_hidden_states)
                    )
                )
            )

            tokens = input_ids[0].cpu().clone().numpy()
            tokens = tokens[1:-1]

            return ModelOutput(bpes, tokens, features)

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(13)] + [f"d{i}" for i in range(13)]


class CodeT5ModelSmall(CodeT5Model):
    def __init__(self, args=[], type="CodeT5_small", model_name="codet5-small"):
        super().__init__(type=type, model_name=model_name)

    @staticmethod
    def get_model(model_type, **kwargs):
        return CodeT5ModelSmall(type=model_type)

    @staticmethod
    def get_embeddings_info() -> List[str]:
        """get identifiers for all embedding layer e.g. e1, e2, e3, ..., d1, d2, d3, ..."""
        return [f"e{i}" for i in range(7)] + [f"d{i}" for i in range(7)]


# class T5ModelBase(CodeT5Model):
#     def __init__(self, args=[], type="T5", model_name="t5-base"):
#         super().__init__(type=type, model_name=model_name)

#     @staticmethod
#     def get_model(model_type, **kwargs):
#         return T5ModelBase(type=model_type)
