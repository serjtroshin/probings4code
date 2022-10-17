import warnings
from typing import List, Optional, Tuple

from .tree_representation import NodeRepresentation

"""
For each bpe token we wind matching AST node (shortest overlaping span principle).
Requirement: all tokens are white-space separated. E.g. no subtoken contains a white.
"""


def match_bpe(tokens_bpe: List[str], tokens: List[str]) -> Optional[dict]:
    """returns an alignment between bpe tokens and code tokens
        NOTE: make sure the code does not contains "▁"

    Args:
        tokens_bpe (List[str]): subtokens
        tokens (List[str]): each token contain at least one bpe token

    Returns:
        dict: bpe -> token
        or None if failed
    """
    # print(">---------")
    # print(tokens_bpe)
    # print(tokens)
    BPE_TOKEN = "▁"
    align = {}
    i = 0
    j = 0
    cur = ""
    while j < len(tokens_bpe):  # all bpe tokens must be matched
        # print(tokens_bpe[j])
        bpe_token = tokens_bpe[j].strip(BPE_TOKEN)
        cur += bpe_token
        # print(cur, tokens[i])
        if not tokens[i].startswith(cur):
            # print("!!!", tokens[i], cur)
            # for i, bpe_token in enumerate(tokens_bpe):
            #     if i in align:
            #         print(bpe_token, tokens[align[i]])
            # input()
            warnings.warn(
                "failed to match (possibly due to some non ascii symbols and bpe parsing errors"
            )
            return None

        align[j] = i  # we are sure that j-th bpe token match i-th token

        if cur == tokens[i]:
            i += 1
            cur = ""
        j += 1

    # if not (i == len(tokens) and j == len(tokens_bpe)):

    return align


def _get_token_spans(sent: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """for each token returns it's span in a sentence
    Args:
        sent (str): white-space delimited sequence of tokens
        tokens (List[str]): substrings of sent (sent.split())

    Returns:
        List[Tuple[int, int]]: list of token spans
    """
    tokens_pos = []
    i = 0
    j = 0
    while i < len(sent):
        assert sent[i : i + len(tokens[j])] == tokens[j], (
            sent[i : i + len(tokens[j])],
            tokens[j],
            sent,
            tokens,
        )
        tokens_pos.append((i, i + len(tokens[j])))
        i += len(tokens[j])
        while i < len(sent) and sent[i].isspace():
            i += 1
        j += 1
    return tokens_pos


def _get_shortest_overlaping_node(
    span: Tuple[int, int], nodes: List[NodeRepresentation]
) -> Tuple[int, NodeRepresentation]:
    """
    Args:
        span (Tuple[int, int]): span of a token
        nodes (List[NodeRepresentation]): AST tree nodes

    Returns:
        NodeRepresentation: a one with the minimal overlaping span
    """

    def _node_span(node) -> Tuple[int, int]:
        # only for one line
        return node.start_point[1], node.end_point[1]

    def _is_overlap(node):
        node_span = _node_span(node.node)
        return span[0] >= node_span[0] and span[1] <= node_span[1]

    def _len_span(node) -> int:
        node_span = _node_span(node.node)
        return node_span[1] - node_span[0]

    min_node_repr = min(filter(_is_overlap, nodes), key=_len_span)
    min_node = min_node_repr.node
    found = True
    while len(min_node.children) > 0 and found:  # try to find the most child also
        found = False
        for child in min_node.children:
            if span[0] >= _node_span(child)[0] and span[1] <= _node_span(child)[1]:
                min_node = child
                found = True
        if not found:
            break
    for it, node in enumerate(nodes):
        if node.node == min_node:
            return it, node


def match_nodes(sent: str, tokens: List[str], nodes: List[NodeRepresentation]) -> dict:
    """
    returns an alignment between tokens in a sent and nodes of AST
    [(tokens_index: range(n), node_index)]
    """
    # values = filter(lambda node: is_identifier(node.node), nodes)
    tokens_pos = _get_token_spans(sent, tokens)
    align = {}
    for i, token_span in enumerate(tokens_pos):
        j, node = _get_shortest_overlaping_node(token_span, nodes)
        align[i] = j
    return align


if __name__ == "__main__":
    from code_parser import CodeParser
    from tree_representation import get_node_representations

    sent = "public static void setState ( State state ) { currentState = state ; }"
    bpe = "▁public ▁static ▁void ▁set State ▁( ▁State ▁state ▁) ▁{ ▁current State ▁= ▁state ▁; ▁}"

    # 1) match bpe tokens with tokens (bpe -> token)
    tokens_bpe = bpe.strip().split()
    tokens = sent.strip().split()

    for pair in match_bpe(tokens_bpe, tokens).items():
        print(tokens_bpe[pair[0]], "\t--->", tokens[pair[1]])

    # 2) match tokens with ast nodes (token -> ast node)

    tokens = sent.strip().split()
    nodes = list(get_node_representations(CodeParser("java")(sent).tree))

    for pair in match_nodes(sent, tokens, nodes).items():
        node = nodes[pair[1]]
        print(tokens[pair[0]], "\t--->", node)

    # 3) bpe -> ast node

    tokens_bpe = bpe.strip().split()
    tokens = sent.strip().split()
    tree = CodeParser("java")(sent)
    nodes = list(get_node_representations(tree.tree))

    bpe2token = match_bpe(tokens_bpe, tokens)
    token2node = match_nodes(sent, tokens, nodes)
    for i, bpe_token in enumerate(tokens_bpe):
        node = nodes[token2node[bpe2token[i]]]
        print(bpe_token, "\t--->", node)
