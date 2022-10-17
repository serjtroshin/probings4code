import enum
import logging
import warnings
from typing import Iterator, List, Tuple

import numpy as np
from termcolor import colored
from tree_sitter import Node

from .aug import (CodeAugmentation, LabeledEdge, LabeledToken, NodeTypes,
                  SentenceInfo, Token, Tree, debuggable,
                  get_identifier_asignment_without_declaration,
                  get_identifier_declarations_with_expr,
                  get_identifier_declarations_without_expr, traverse_tree)


class UnidentifiedVariablesHard(CodeAugmentation):
    def __init__(self, type="undeclared"):
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
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:
        """code augementation (java)
        for declarated (somewere) identifier var, insert `print(var); on the right of declaration`
        was var declarated before usage?

        Returns:
            Tuple[List[str], Dict[str, int]]: augmented code and labels for tokens
            labels:
                0 -- unused token
                1 -- declarated identifier
                2 -- undeclarated identifier
                3 -- target declarator
        """
        decl_token_ids = get_identifier_declarations_with_expr(
            tokens, ast, only_int_types=False
        )
        decl_token_ids.extend(
            get_identifier_declarations_without_expr(tokens, ast, only_int_types=False)
        )

        if len(decl_token_ids) == 0:
            warnings.warn("no declarations")
            return

        starts_of_block = []  # {
        for it, token in enumerate(tokens):
            if token.node.type == "{":
                starts_of_block.append(it)

        ends_of_block = []  # ; }    or   } }
        for it, token in enumerate(tokens):
            if (
                token.node.type == ";"
                and it + 1 < len(tokens)
                and tokens[it + 1].node.type == "}"
            ):
                ends_of_block.append(it)
            elif (
                token.node.type == "}"
                and it + 1 < len(tokens)
                and tokens[it + 1].node.type == "}"
            ):
                ends_of_block.append(it)

        identifier_pos, identifier_block = decl_token_ids[
            np.random.randint(0, len(decl_token_ids))
        ]  # target declaration

        possible_positions_to_insert = starts_of_block + ends_of_block

        possible_positions_to_insert = list(
            filter(lambda pos: pos > identifier_block[-1], possible_positions_to_insert)
        )
        # remain only those that are on the right of the declaration
        # otherwise the task is too simple

        if len(possible_positions_to_insert) == 0:
            warnings.warn("no positions to insert statement")
            return

        pos_to_insert = possible_positions_to_insert[
            np.random.randint(0, len(possible_positions_to_insert))
        ]

        type_of_ident = tokens[identifier_block[0]].node.type
        identifier_value = tokens[identifier_pos].value

        # if len(list(decl for decl in decl_token_ids if tokens[decl[0]].value == identifier_value)) != 1:
        #     # select only snippets with a single identifier declaration
        #     return

        all_identifier_declarations = list(
            decl for decl in decl_token_ids if tokens[decl[0]].value == identifier_value
        )
        all_blocks = [block for ident, block in all_identifier_declarations]
        all_identifier_pos = set(pos for block in all_blocks for pos in block)

        new_code_tokens = []
        TARGET = "TARGET_IDENT_"
        INSERT = "NEW_IDENT_TO_INSERT_"
        try:
            for it, token in enumerate(tokens):
                if (
                    token.node.type == NodeTypes.Identifier
                    and token.value == identifier_value
                    and it in all_identifier_pos
                ):
                    new_code_tokens.append(TARGET)
                else:
                    new_code_tokens.append(token.value)

                if it == pos_to_insert:
                    new_code_tokens.append("System")
                    new_code_tokens.append(".")
                    new_code_tokens.append("out")
                    new_code_tokens.append(".")
                    new_code_tokens.append("println")
                    new_code_tokens.append("(")
                    new_code_tokens.append(INSERT)
                    new_code_tokens.append(")")
                    new_code_tokens.append(";")
                    # new_code_tokens.append("=")
                    # new_code_tokens.append(FILL_VALUE[type_of_ident])
        except KeyError as e:
            print(e)
            print(decl_token_ids)
            return

        new_code = " ".join(new_code_tokens)

        # print("new_code", new_code)

        new_code_tokens, _ = self.process(new_code)
        if "ERROR" in [tok.node.type for tok in new_code_tokens]:
            warnings.warn("ERROR")
            logging.info(f"new_code: {new_code}")
            return

        target_ids = []
        insert_id = None
        for it, token in enumerate(new_code_tokens):
            if token.value == TARGET:
                target_ids.append(it)
            if token.value == INSERT:
                insert_id = it
        if len(target_ids) == 0 or insert_id is None:
            warnings.warn("not found")
            return

        is_visible = any(
            [
                is_left_visible_to_right(new_code_tokens, target_id, insert_id)
                for target_id in target_ids
            ]
        )

        new_tokens = []
        for it, token in enumerate(new_code_tokens):
            if it in target_ids:
                new_tokens.append(LabeledToken(tokens[identifier_pos].value, 3))
            elif it == insert_id:
                new_tokens.append(
                    LabeledToken(tokens[identifier_pos].value, 1 if is_visible else 2)
                )
            else:
                new_tokens.append(LabeledToken(token.value, 0))

        yield new_tokens, [], SentenceInfo(None)


def get_root_path(node: Node):
    # path from root to the node
    nodes = []
    while True:
        nodes.append(node)
        node = node.parent
        if node is None:
            break
    return nodes[::-1]


def is_left_visible_to_right(tokens: List[Token], left: int, right: int):
    left_node = tokens[left].node
    right_node = tokens[right].node
    assert left_node is not None
    assert right_node is not None
    left_parents = list(filter(lambda x: x.type == "block", get_root_path(left_node)))
    right_parents = list(filter(lambda x: x.type == "block", get_root_path(right_node)))

    while len(right_parents) > 0 and (
        len(left_parents) == 0
        or len(left_parents) > 0
        and right_parents[-1] != left_parents[-1]
    ):
        right_parents.pop()
    while (
        len(right_parents) > 0
        and len(left_parents) > 0
        and right_parents[-1] == left_parents[-1]
    ):
        right_parents.pop()
        left_parents.pop()
    if not (len(left_parents) == 0 and len(right_parents) == 0):
        return False
    return left < right


def test_is_visible(code):
    # print("------ test -------")
    # # print(colored(code, "red"))
    # aug = CodeAugmentation()

    # ast = aug.code_parser(code).tree
    # tokens, ast = aug.process(code)

    # for ident_decl, block_decl in get_identifier_declarations_with_expr(tokens, ast):
    #     for ident_ass, block_ass in get_identifier_asignment_without_declaration(tokens, ast):
    #         for it, token in enumerate(tokens):
    #             if it in block_decl:
    #                 print(colored(token.value, "red"), end=" ")
    #             elif it in block_ass:
    #                 print(colored(token.value, "green"), end=" ")
    #             else:
    #                 print(token.value, end=" ")
    #         # print("left", tokens[ident_decl], [tokens[i].value for i in block_decl])
    #         # print("right", tokens[ident_ass], [tokens[i].value for i in block_ass])
    #         print("is visible ?", is_left_visible_to_right(tokens, ident_decl, ident_ass))

    print("----- test aug ----")
    aug = UnidentifiedVariablesHard()
    tokens, ast = aug.process(code)
    for new_tokens, _ in aug(tokens, ast):
        # print(f"----------------- {labels[sent_info.label]} --------------")
        for x in new_tokens:
            print(colored(x.value, colors[x.label]), end=" ")
        print()


if __name__ == "__main__":
    colors = {0: None, 1: "green", 2: "red", 3: "blue"}
    # labels = {0: "original", 1: "semantic preserving", 2: "semantic non-preserving"}
    np.random.seed(9)

    aug = UnidentifiedVariablesHard()

    code = "int result = 1 ; if ( result == 0 ) { int x = 0 ; } else { int y = 0 ; } x = 0 ; int z ;"
    test_is_visible(code)

    code = "x = 0 ; int x = 0 ;"
    test_is_visible(code)

    code = "int x = 0 ; x = 0 ;"
    test_is_visible(code)

    code = "int x = 0 ; { x = 0 ; }"
    test_is_visible(code)

    code = "{ int x = 0 ; } x = 0 ;"
    test_is_visible(code)

    code = "{ int x = 0 ; } { x = 0 ; }"
    test_is_visible(code)

    code = "{ for ( int i = 0; i < 10 ; i ++ ) { int x = 0 ; if ( x == 1 ) { x = 0 ; } { } } }"
    test_is_visible(code)

    code = "{ for ( int i = 0; i < 10 ; i ++ ) { int x = 0 ; if ( x == 1 ) { x = 0 ; } } { } }"
    test_is_visible(code)

    code = "{ int x = 0 ; } { int x = 0 ; }"
    test_is_visible(code)

    code = "{ int x = 0 ; } { int x = 0 ; }"
    test_is_visible(code)

    code = "{ int x = 0 ; } { int x = 0 ; }"
    test_is_visible(code)

    code = "{ for ( int i = 0; i < 10 ; i ++ ) { float x = 0.0 ; if ( x == 1 ) { x = 0 ; } } { } }"
    test_is_visible(code)

    code = 'String s = " aaa " ; '
    for node in traverse_tree(aug.process(code)[1]):
        print(node)
    test_is_visible(code)
