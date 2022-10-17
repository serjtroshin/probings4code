from collections import defaultdict
from multiprocessing.sharedctypes import Value
from typing import Any, Iterator, List, Optional, Tuple

from src.models.model import ModelOutput
from src.struct_probing.utils.code_parser import CodeParser, traverse_tree
from src.struct_probing.utils.match_tokens import (match_bpe,
                                                              match_nodes)
from src.struct_probing.utils.tree_representation import \
    get_node_representations
from termcolor import colored
from tree_sitter import Node, Tree

from .constants import NodeTypes, RequiredEmbeddings, TokenTypes


def debuggable(method):
    colors = {None: None, 1: "green", 2: "red", 3: "blue", 4: "yellow"}

    def debug(
        self, tokens: List[Token], ast: Tree, prob_err=0.3, debug=False
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:
        result = method(self, tokens, ast)
        tok2col = {}
        i = 1
        for new_tokens, edges, sent_info in result:
            if debug:
                if sent_info.label is not None:
                    print(f">> {sent_info.label} <<")
                for x in new_tokens:
                    if x.label is None:
                        print(x.value, end=" ")
                    else:
                        label = x.label
                        if not label in tok2col:
                            assert i <= 4
                            tok2col[label] = i
                            i += 1
                        print(colored(x.value, colors[tok2col[x.label]]), end=" ")
                print()
                for edge in edges:
                    print(tokens[edge.first], tokens[edge.second], edge.label)
                print()
            yield new_tokens, edges, sent_info

    return debug


class Token:
    def __init__(self, value: str, node: Node):
        self.value = value
        self.node = node

    def __str__(self):
        return self.value


class LabeledToken:
    def __init__(self, value: str, label: Any):
        self.value = value
        self.label = label

    def __str__(self):
        return self.value

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(obj):
        return LabeledToken(obj["value"], obj["label"])


class LabeledTokenTyped:
    # preferable
    def __init__(self, value: str, type: str, label: Any):
        self.value = value
        self.type = type
        self.label = label

    def __str__(self):
        return self.value

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(obj):
        return LabeledTokenTyped(obj["value"], obj["label"], obj["type"])


class LabeledEdge:
    def __init__(self, first: int, second: int, label: Any):
        self.first = first
        self.second = second
        self.label = label

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(obj):
        return LabeledEdge(obj["first"], obj["second"], obj["label"])


class SentenceInfo:
    def __init__(self, label: Any):
        self.label = label

    def to_json(self):
        return self.__dict__


class CodeAugmentation:
    def __init__(self, type="identity", lang="java"):
        self._type = type

        self.code_parser = CodeParser(lang=lang)

    @property
    def type(self) -> str:
        return self._type

    def required_dataset(self) -> str:
        return None

    def required_embeddings(self) -> RequiredEmbeddings:
        return RequiredEmbeddings.DUMMY

    def process(self, code: str) -> Tuple[List[Token], Tree]:
        """[summary]

        Args:
            code (str): [description]

        Returns:
            Tuple[List[Token], Tree]: (leave tokens, ast,)
        """

        code = code.strip().replace("▁", "_")  # to prevent error with bpe
        # print("code:", code)

        code_tokens: List[str] = code.split()
        ast = self.code_parser(code).tree

        nodes = list(get_node_representations(ast))

        result_tokens: List[Token] = []
        for pair in match_nodes(code, code_tokens, nodes).items():
            node = nodes[pair[1]]
            # print(code_tokens[pair[0]], "\t--->", node)
            result_tokens.append(Token(code_tokens[pair[0]], node.node))

        return result_tokens, ast

    def __call__(
        self, tokens: List[Token], ast: Tree, debug=False
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:
        """code augmentation

        Returns:
            List[LabeledToken]: augmented tokens and additional labels
        """
        yield [LabeledToken(token.value, None) for token in tokens], [], SentenceInfo(
            None
        )

    def embeddings_hook(
        self,
        model_output: ModelOutput,
        labeled_bpe: Optional[List[Any]],
        labeled_edges: List[LabeledEdge],
    ) -> Tuple[ModelOutput, Optional[List[Any]], LabeledEdge]:
        # by default do nothing
        return model_output, labeled_bpe, labeled_edges

    def bpe_labels(
        self, tokens: List[LabeledToken], bpes: List[str]
    ) -> Optional[List[Any]]:
        bpe2token = match_bpe(bpes, [t.value for t in tokens])
        if bpe2token is None:
            return None
        return [tokens[bpe2token[i]].label for i, _ in enumerate(bpes)]

    def bpe_edges(
        self, edges: List[LabeledEdge], tokens: List[LabeledToken], bpes: List[str]
    ) -> List[LabeledEdge]:
        bpe2token = match_bpe(bpes, [t.value for t in tokens])
        token2bpes = defaultdict(list)
        for bpe_idx, tok_idx in bpe2token.items():
            token2bpes[tok_idx].append(bpe_idx)
        token2first_bpe = {
            tok_idx: (min(bpe_idxs), max(bpe_idxs))
            for tok_idx, bpe_idxs in token2bpes.items()
        }
        new_edges = []
        for edge in edges:
            if not edge.first in token2first_bpe or not edge.second in token2first_bpe:
                continue
            if edge.label != 0:
                pass
                # print(bpes[token2first_bpe[edge.first]], bpes[token2first_bpe[edge.second]], edge.label)
            new_edges.append(
                LabeledEdge(
                    token2first_bpe[edge.first],
                    token2first_bpe[edge.second],
                    edge.label,
                )
            )
        # for edge in edges[:10]:
        #     print("TOKEN BPE", tokens[edge.first], tokens[edge.second], edge.label)
        # for edge in new_edges[:10]:
        #     print("EDGE BPE", bpes[edge.first], bpes[edge.second], edge.label)
        # input()
        return new_edges


def find_node_pos_in_tokens(tokens: List[Token], node: Node) -> Optional[int]:
    for it, token in enumerate(tokens):
        if token.node == node:
            return it
    return None


def find_all_token_pos_in_node(tokens: List[Token], node: Node) -> List[int]:
    nodes = list(map(lambda x: x.node, get_node_representations(node)))
    block = []
    for it, token in enumerate(tokens):
        if token.node in nodes:
            block.append(it)
    return block


def get_identifier_declarations_without_expr(
    tokens: List[Token], ast: Tree, only_int_types=True
) -> List[Tuple[int, List[int]]]:
    """
    int var ;
    Returns:
        List[Tuple[str, List[int, int]]]: [position of var token, variable declarator positions]
    """
    variable_declarators = list(
        filter(
            lambda x: x.type == NodeTypes.LocalVariableDeclaration, traverse_tree(ast)
        )
    )
    variable_declarators = list(
        filter(lambda x: x.parent.type != NodeTypes.ForStatement, variable_declarators)
    )
    if only_int_types:
        variable_declarators = list(
            filter(
                lambda x: x.children[0].type
                in (NodeTypes.IntegralType, NodeTypes.FloatingPointType)
                and x.children[1].type == NodeTypes.VariableDeclarator
                and x.children[2].type == ";",
                variable_declarators,
            )
        )  # int var ;
    else:
        variable_declarators = list(
            filter(
                lambda x: x.children[1].type == NodeTypes.VariableDeclarator
                and x.children[2].type == ";",
                variable_declarators,
            )
        )
    variable_declarators = list(
        filter(
            lambda x: len(x.children[1].children) == 1
            and x.children[1].children[0].type == TokenTypes.Identifier,
            variable_declarators,
        )
    )

    blocks = []
    for declarator in variable_declarators:
        variable_name_node = declarator.children[1].children[0]
        identifier = find_node_pos_in_tokens(tokens, variable_name_node)
        if identifier is None:
            continue
        declarator_tokens = find_all_token_pos_in_node(tokens, declarator)
        blocks.append((identifier, declarator_tokens))
    return blocks


def get_identifier_declarations_with_expr(
    tokens: List[Token], ast: Tree, only_int_types=True
) -> List[Tuple[int, List[int]]]:
    """
    int var = [expression]
    Returns:
        List[Tuple[str, List[int, int]]]: [position of var token, variable declarator positions, type]
    """
    variable_declarators = list(
        filter(
            lambda x: x.type == NodeTypes.LocalVariableDeclaration, traverse_tree(ast)
        )
    )
    variable_declarators = list(
        filter(lambda x: x.parent.type != NodeTypes.ForStatement, variable_declarators)
    )
    if only_int_types:
        variable_declarators = list(
            filter(
                lambda x: x.children[0].type
                in (NodeTypes.IntegralType, NodeTypes.FloatingPointType)
                and x.children[1].type == NodeTypes.VariableDeclarator,
                variable_declarators,
            )
        )  # int declarators
    else:
        variable_declarators = list(
            filter(
                lambda x: x.children[1].type == NodeTypes.VariableDeclarator,
                variable_declarators,
            )
        )
    # with init
    variable_declarators = list(
        filter(
            lambda x: len(x.children[1].children) > 2
            and x.children[1].children[0].type == TokenTypes.Identifier
            and x.children[1].children[1].type == "=",
            variable_declarators,
        )
    )

    blocks = []
    for declarator in variable_declarators:
        variable_name_node = declarator.children[1].children[0]
        identifier = find_node_pos_in_tokens(tokens, variable_name_node)
        if identifier is None:
            continue
        declarator_tokens = find_all_token_pos_in_node(tokens, declarator)
        blocks.append((identifier, declarator_tokens))
    return blocks


def get_identifier_asignment_without_declaration(
    tokens: List[Token], ast: Tree
) -> List[Tuple[int, List[int]]]:
    """
    var = [expression]
    Returns:
        List[Tuple[str, List[int, int]]]: [position of var token, variable declarator token positions]
    """
    statements = list(
        filter(lambda x: x.type == NodeTypes.ExpressionStatement, traverse_tree(ast))
    )
    statements = list(
        filter(lambda x: x.parent.type != NodeTypes.ForStatement, statements)
    )
    statements = list(
        filter(
            lambda x: x.children[0].type == NodeTypes.AssignmentExpression
            and x.children[1].type == ";",
            statements,
        )
    )
    statements = list(
        filter(
            lambda x: x.children[0].children[0].type == NodeTypes.Identifier
            and x.children[0].children[1].type == "=",
            statements,
        )
    )
    blocks = []
    for expr_statement in statements:
        variable_name_node = expr_statement.children[0].children[0]
        identifier = find_node_pos_in_tokens(tokens, variable_name_node)
        if identifier is None:
            continue
        declarator_tokens = find_all_token_pos_in_node(tokens, expr_statement)
        blocks.append((identifier, declarator_tokens))
    return blocks


if __name__ == "__main__":
    print("------------")
    code = "int result = 0 + 0 ; float x = 1 ; File [ ] files = dir . getFiles ( ) ; int k = 0 ;"
    aug = CodeAugmentation()
    tokens, ast = aug.process(code)

    for ident, block in get_identifier_declarations_with_expr(tokens, ast):
        print(tokens[ident].value, [tokens[i].value for i in block])

    # result ['int', 'result', '=', '0', '+', '0', ';']
    # k ['int', 'k', '=', '0', ';']

    print("------------")
    code = "for ( int i = 0 ; i < 10 ; i ++ ) ; int k = 0 ;"
    aug = CodeAugmentation()
    tokens, ast = aug.process(code)

    for ident, block in get_identifier_declarations_with_expr(tokens, ast):
        print(tokens[ident].value, [tokens[i].value for i in block])
