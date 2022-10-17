# from typing import Iterator, List, Tuple

# import numpy as np

# from .aug import (CodeAugmentation, LabeledToken, SentenceInfo,
#                   Token, Tree, get_identifier_declarations_with_expr,
#                   get_identifier_declarations_without_expr)


# def debuggable(method):
#     from termcolor import colored

#     colors = {0: None, 1: "green", 2: "red"}

#     def debug(
#         self, tokens: List[Token], ast: Tree, prob_err=0.3, debug=False
#     ) -> Iterator[Tuple[List[LabeledToken], SentenceInfo]]:
#         result = method(self, tokens, ast)
#         for new_tokens, sent_info in result:
#             if debug:
#                 for x in new_tokens:
#                     print(colored(x.value, colors[x.label]), end=" ")
#                 print()
#             yield new_tokens, sent_info

#     return debug


# class UnidentifiedVariables(CodeAugmentation):
#     def __init__(self, type="unidentified_var"):
#         super().__init__(type)
#         self._type = type

#         self.brackets = "{}()[]<>"
#         others = {}
#         for i in range(len(self.brackets)):
#             others[self.brackets[i]] = self.brackets[:i] + self.brackets[i + 1 :]
#         self.others = others
#         self.ln_others = len(self.brackets) - 1

#     @debuggable
#     def __call__(
#         self, tokens: List[Token], ast: Tree, prob_err=0.3
#     ) -> Iterator[Tuple[List[LabeledToken], SentenceInfo]]:
#         """code augementation (java)
#         we select all unique identifiers and then for some int identifier var we maybe insert
#         'var = 0' right after some ';' obtaining a valid code snippet or at the left of declaration,
#         obtaing wrong code.
#         Next, the label for a inserted identifier is 'was it declarated beforehand in the code snippet?'

#         Args:
#             prob_err (float): probability of code insertion

#         Returns:
#             Tuple[List[str], Dict[str, int]]: augmented code and labels for tokens
#             labels:
#                 0 -- unused token
#                 1 -- declarated identifier
#                 2 -- undeclarated identifier
#         """
#         new_tokens = []

#         decl_token_ids = get_identifier_declarations_with_expr(tokens, ast)
#         decl_token_ids.extend(get_identifier_declarations_without_expr(tokens, ast))

#         if len(decl_token_ids) == 0:
#             return

#         start_token2token_ids = {ids[1][0]: ids for ids in decl_token_ids}
#         end_token2token_ids = {ids[1][-1]: ids for ids in decl_token_ids}

#         try:
#             for it, token in enumerate(tokens):
#                 if it in start_token2token_ids:
#                     identifier_block = start_token2token_ids[it][1]
#                     type_of_ident = tokens[identifier_block[0]].node.type
#                     if np.random.rand() < prob_err:
#                         identifier_pos = start_token2token_ids[it][0]
#                         identifier_value = tokens[identifier_pos].value

#                         new_tokens.append(LabeledToken(identifier_value, 2))
#                         # 1 if declarated, 2 if not declarated
#                         new_tokens.append(LabeledToken("=", 0))
#                         new_tokens.append(LabeledToken(FILL_VALUE[type_of_ident], 0))
#                         new_tokens.append(LabeledToken(";", 0))

#                 new_tokens.append(LabeledToken(token.value, 0))

#                 if it in end_token2token_ids:
#                     identifier_block = end_token2token_ids[it][1]
#                     type_of_ident = tokens[identifier_block[0]].node.type
#                     if np.random.rand() < prob_err:
#                         identifier_pos = end_token2token_ids[it][0]
#                         identifier_value = tokens[identifier_pos].value

#                         new_tokens.append(LabeledToken(identifier_value, 1))
#                         # 1 if declarated, 2 if not declarated
#                         new_tokens.append(LabeledToken("=", 0))
#                         new_tokens.append(LabeledToken(FILL_VALUE[type_of_ident], 0))
#                         new_tokens.append(LabeledToken(";", 0))
#         except KeyError as e:
#             print(e)
#             print(decl_token_ids)
#             return

#         yield new_tokens, SentenceInfo(None)


# if __name__ == "__main__":
#     from termcolor import colored

#     colors = {0: None, 1: "green", 2: "red"}
#     # labels = {0: "original", 1: "semantic preserving", 2: "semantic non-preserving"}
#     np.random.seed(3)

#     code = "int result = 1 ; float x = result + 0 ; File [ ] files = dir . getFiles ( ) ; int k = 0 ;"
#     aug = UnidentifiedVariables()
#     tokens, ast = aug.process(code)

#     for new_tokens, sent_info in aug(tokens, ast):
#         # print(f"----------------- {labels[sent_info.label]} --------------")
#         for x in new_tokens:
#             print(colored(x.value, colors[x.label]), end=" ")
#         print()
