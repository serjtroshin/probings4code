from typing import Iterator, List, Tuple

import numpy as np

from .aug import (CodeAugmentation, LabeledToken, SentenceInfo, Token,
                  TokenTypes, Tree, debuggable,
                  get_identifier_declarations_with_expr)


class Labels:
    Original = 0
    SemanticPreserving = 1
    NonSemanticPreserving = 2


# def debuggable(method):
#     from termcolor import colored

#     colors = {0: None, 1: "green", 2: "blue"}
#     labels = {0: "original", 1: "semantic preserving", 2: "semantic non-preserving"}

#     def debug(
#         self, tokens: List[Token], ast: Tree, prob_err=0.15, debug=False
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


class SemanticVariableInsert(CodeAugmentation):
    def __init__(self, type="variable_insert"):
        super().__init__(type)
        self._type = type

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], SentenceInfo]]:
        """code augementation (java)
        augmentations:
        1. Semantic preserving:
            - insert 'var = 0' before var = [expresion]
            - insert new_var = 0 after var = [expresion]
        2. Semantic non-preserving:
            - insert 'var = 0' after var = [expresion]
        Note: if [expresion] == 0, all expressions will be preserving

        Returns:
            Tuple[List[str], Dict[str, int]]: augmented code and labels for tokens
            SentenceInfo:
                0 -- orig sentence
                1 -- semantic preserving
                2 -- non semantic preserving
            LabeledToken:
                0 - old tokens irrelevant
                1 - new tokens
                2 - old tokens relevant
        """
        decl_token_ids = get_identifier_declarations_with_expr(tokens, ast)
        if len(decl_token_ids) == 0:
            return

        # choose random decl_token_ids here (alternative to randomizing for each variant)
        target_position = decl_token_ids[np.random.randint(0, len(decl_token_ids))]
        decl_token_ids = [target_position]

        # all_identifiers = np.unique([tokens[ident_pos].value for ident_pos, _ in decl_token_ids])  # all declared identifiers
        # start_token2token_ids = {ids[1][0] : ids for ids in decl_token_ids}
        # end_token2token_ids = {ids[1][-1] : ids for ids in decl_token_ids}

        # orig sentence
        yield [
            LabeledToken(tok.value, 0)
            if not i in decl_token_ids[0][1]
            else LabeledToken(tok.value, 2)
            for i, tok in enumerate(tokens)
        ], SentenceInfo(Labels.Original)

        # 1) insert 'var = 0' before var = [expresion]
        new_tokens = []
        target_position = decl_token_ids[np.random.randint(0, len(decl_token_ids))]
        iden_pos, block = target_position
        target_value = tokens[iden_pos].value
        start_pos = block[0]

        for it, token in enumerate(tokens):
            new_tokens.append(LabeledToken(token.value, 0))  # type
            if it == start_pos:
                new_tokens.append(LabeledToken(target_value, 1))
                new_tokens.append(LabeledToken("=", 1))
                new_tokens.append(LabeledToken("0", 1))
                new_tokens.append(LabeledToken(";", 1))

        yield new_tokens, SentenceInfo(Labels.SemanticPreserving)

        # 2) insert new_var = 0 after var = [expresion]
        # all_identifiers = np.unique([tokens[ident_pos].value for ident_pos, _ in decl_token_ids])  # all declared identifiers

        new_tokens = []
        target_position = decl_token_ids[np.random.randint(0, len(decl_token_ids))]
        iden_pos, block = target_position
        target_value = tokens[iden_pos].value
        end_pos = block[-1]

        for it, token in enumerate(tokens):
            new_tokens.append(LabeledToken(token.value, 0))
            if it == end_pos:
                new_tokens.append(LabeledToken("int", 1))
                new_tokens.append(LabeledToken(f"new_{target_value}", 1))
                new_tokens.append(LabeledToken("=", 1))
                new_tokens.append(LabeledToken("0", 1))
                new_tokens.append(LabeledToken(";", 1))

        yield new_tokens, SentenceInfo(Labels.SemanticPreserving)

        # 3) insert 'var = 0' after var = [expresion]
        new_tokens = []
        target_position = decl_token_ids[np.random.randint(0, len(decl_token_ids))]
        iden_pos, block = target_position
        target_value = tokens[iden_pos].value
        end_pos = block[-1]
        expression_values = block[1:]
        # we do not consider other possible expressions here such as zero()
        is_value_zero = False
        if len(expression_values) == 4:  # ["=", "0", ";"]
            if tokens[expression_values[-2]].value == "0":
                is_value_zero = True

        for it, token in enumerate(tokens):
            new_tokens.append(LabeledToken(token.value, 0))
            if it == end_pos:
                new_tokens.append(LabeledToken(f"{target_value}", 1))
                new_tokens.append(LabeledToken("=", 1))
                new_tokens.append(LabeledToken("0", 1))
                new_tokens.append(LabeledToken(";", 1))

        yield new_tokens, SentenceInfo(
            Labels.NonSemanticPreserving
            if not is_value_zero
            else Labels.SemanticPreserving
        )  # error


if __name__ == "__main__":
    from termcolor import colored

    colors = {0: None, 1: "green"}
    labels = {0: "original", 1: "semantic preserving", 2: "semantic non-preserving"}
    np.random.seed(3)

    code = "int result = 1 ; float x = result + 0 ; File [ ] files = dir . getFiles ( ) ; int k = 0 ;"
    aug = SemanticVariableInsert()
    tokens, ast = aug.process(code)

    for new_tokens, sent_info in aug(tokens, ast):
        print(f"----------------- {labels[sent_info.label]} --------------")
        for x in new_tokens:
            print(colored(x.value, colors[x.label]), end=" ")
        print()

    print("--------------")
    code = "float x = result + 0 ; File [ ] files = dir . getFiles ( ) ; int k = 1 ;"
    aug = SemanticVariableInsert()
    tokens, ast = aug.process(code)

    for new_tokens, sent_info in aug(tokens, ast):
        print(f"----------------- {labels[sent_info.label]} --------------")
        for x in new_tokens:
            print(colored(x.value, colors[x.label]), end=" ")
        print()
