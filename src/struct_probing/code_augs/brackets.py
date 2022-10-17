import random
from typing import Iterator, List, Tuple

import numpy as np

from .aug import (CodeAugmentation, LabeledEdge, LabeledToken, SentenceInfo,
                  Token, Tree, debuggable)

# def debuggable(method):
#     from termcolor import colored

#     colors = {0: None, 1: "green", 2: "red"}
#     labels = None

#     def debug(
#         self, tokens: List[Token], ast: Tree, prob_err=0.30, debug=False
#     ) -> Iterator[Tuple[List[LabeledToken], SentenceInfo]]:
#         result = method(self, tokens, ast)
#         for new_tokens, sent_info in result:
#             if debug:
#                 if sent_info.label is not None:
#                     print(f"----------------- {labels[sent_info.label]} --------------")
#                 for x in new_tokens:
#                     print(colored(x.value, colors[x.label]), end=" ")
#                 print()
#             yield new_tokens, sent_info

#     return debug


class BracketsCodeAugmentation(CodeAugmentation):
    def __init__(self, type="brackets"):
        super().__init__(type)
        self._type = type

        self.brackets = "{}()[]<>"
        others = {}
        for i in range(len(self.brackets)):
            others[self.brackets[i]] = self.brackets[:i] + self.brackets[i + 1 :]
        self.others = others
        self.ln_others = len(self.brackets) - 1

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree, prob_err=0.30
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:
        """code augmentation

        Args:
            prob_err (float): probability of error insertion

        Returns:
            Tuple[List[str], Dict[str, int]]: augmented code and labels for tokens
            labels:
                0 -- unused token
                1 -- unchanged bracket
                2 -- changed bracket
        """
        new_tokens = []
        for token in tokens:
            token_type = token.node.type
            old_token = token.value
            if (
                token_type not in self.brackets
            ):  # if a bracket token in inside string expression, skip it (label=0)
                new_tokens.append(LabeledToken(old_token, 0))
                continue

            token_chars = [old_token[i] for i in range(len(old_token))]

            # find brackets
            pos = []
            for i, c in enumerate(token_chars):
                if c in self.brackets:
                    pos.append(i)

            # replace random brackets
            coin = np.random.uniform(size=len(pos))
            is_changed = coin < prob_err

            for i, p in enumerate(pos):
                # for each position of a bracket if it should be changed, replace it
                if is_changed[i]:
                    ind = random.randint(0, self.ln_others - 1)
                    replacement = self.others[token_chars[p]][ind]
                    token_chars[p] = replacement

            new_token = "".join(token_chars)
            new_tokens.append(
                LabeledToken(new_token, int(new_token != old_token) + 1)
            )  # 2 if added error, else 1
        yield new_tokens, [], SentenceInfo(None)


if __name__ == "__main__":
    np.random.seed(3)
    code = "int result = 1 ; float x = result + 0 ; File [ ] files = dir . getFiles ( ) ; int k = 0 ;"
    aug = BracketsCodeAugmentation()
    tokens, ast = aug.process(code)

    for new_tokens, sent_info in aug(tokens, ast, debug=True):
        pass
        # # print(f"----------------- {labels[sent_info.label]} --------------")
        # for x in new_tokens:
        #     print(colored(x.value, colors[x.label]), end=" ")
        # print()
