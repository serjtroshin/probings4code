import logging
import warnings
from typing import Iterator, List, Tuple

import numpy as np

from .aug import (CodeAugmentation, CodeParser, LabeledEdge, LabeledToken,
                  SentenceInfo, Token, Tree, debuggable)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from evaluation.CodeBLEU.parser import (DFG_csharp, DFG_go, DFG_java,
                                        DFG_javascript, DFG_php, DFG_python,
                                        DFG_ruby, index_to_code_token,
                                        remove_comments_and_docstrings,
                                        tree_to_token_index,
                                        tree_to_variable_index)

lang = "java"
parser = [CodeParser(lang).parser, DFG_java]


def DFG_java(root_node, index_to_code, states):
    assignment = ["assignment_expression"]
    def_statement = ["variable_declarator"]
    increment_statement = ["update_expression"]
    if_statement = ["if_statement", "else"]
    for_statement = ["for_statement"]
    enhanced_for_statement = ["enhanced_for_statement"]
    while_statement = ["while_statement"]
    do_first_statement = []
    states = states.copy()
    if (
        len(root_node.children) == 0 or root_node.type == "string"
    ) and root_node.type != "comment":
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, "comesFrom", [code], states[code].copy())], states
        else:
            if root_node.type == "identifier":
                states[code] = [idx]
            return [(code, idx, "comesFrom", [], [])], states
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name("name")
        value = root_node.child_by_field_name("value")
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, "comesFrom", [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "comesFrom", [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name("left")
        right_nodes = root_node.child_by_field_name("right")
        DFG = []
        temp, states = DFG_java(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if "else" in root_node.type:
            tag = True
        for child in root_node.children:
            if "else" in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_java(child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_java(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_java(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
            elif child.type == "local_variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name("name")
        value = root_node.child_by_field_name("value")
        body = root_node.child_by_field_name("body")
        DFG = []
        for i in range(2):
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = DFG_java(body, index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def get_data_flow(tokens, ast):
    # try:
    # logger.debug(f"> tree:  {[node for node in traverse_tree(tree)]}")
    code = " ".join([t.value for t in tokens])
    #     print(code)
    root_node = ast.root_node
    #     print("tokens:", tokens)
    tokens_index = tree_to_token_index(root_node)
    #     print("tokens_index", tokens_index)

    code = code.split("\n")
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    #     print("code_tokens", code_tokens)

    tokens_index2my_tokens = {}
    for i, index in enumerate(tokens_index):
        tokens_index2my_tokens[i] = None
        for idx, token in enumerate(tokens):
            if (token.node.start_point, token.node.end_point) == index:
                tokens_index2my_tokens[i] = idx
    #     print("tokens_index2my_tokens", tokens_index2my_tokens)

    # plbart part
    index_to_code = {}  # start, end -> index of token
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    # try:
    DFG, _ = DFG_java(root_node, index_to_code, {})
    # except:
    #     DFG = []
    DFG = sorted(DFG, key=lambda x: x[1])
    #     logger.debug(f"> DFG:  {DFG}")
    indexs = set()
    for d in DFG:
        if len(d[-1]) != 0:
            indexs.add(d[1])
        for x in d[-1]:
            indexs.add(x)
    new_DFG = []
    for d in DFG:
        if d[1] in indexs:
            new_DFG.append(d)
    #     logger.debug(f"> new_DFG:  {new_DFG}")
    codes = code_tokens
    dfg = new_DFG

    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (
                d[0],
                d[1],
                d[2],
                list(set(dic[d[1]][3] + d[3])),
                list(set(dic[d[1]][4] + d[4])),
            )
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    #     logger.debug(f"> final DFG:  {new_DFG}")

    var_edges = []  # idx, from_idx, type
    for token_str, idx, edge_type, from_var, from_idx in new_DFG:
        if len(from_var) > 0:
            if (
                tokens_index2my_tokens[idx] is None
                or tokens_index2my_tokens[from_idx[0]] is None
            ):
                warnings.warn("tokens_index2my_tokens[idx] is None")
                continue
            var_edges.append(
                (
                    tokens_index2my_tokens[idx],
                    tokens_index2my_tokens[from_idx[0]],
                    edge_type,
                )
            )
    return var_edges


class Labels:
    NoEdge = "NoEdge"
    comesFrom = "comesFrom"
    computedFrom = "computedFrom"


# def debuggable(method):
#     from termcolor import colored

#     colors = {0: None, 1: "green"}

#     def debug(
#         self, tokens: List[Token], ast: Tree, prob_err=0.15, debug=False
#     ) -> Iterator[Tuple[List[LabeledToken], SentenceInfo]]:
#         result = method(self, tokens, ast)
#         for new_tokens, sent_info in result:
#             if debug:
#                 if sent_info.label is not None:
#                     print(f"----------------- {sent_info.label} --------------")
#                 for x in new_tokens:
#                     print(colored(x.value, colors[x.label]), end=" ")
#                 print()
#             yield new_tokens, sent_info

#     return debug


class IdentityWithDataFlowGraph(CodeAugmentation):
    def __init__(self, type="dfg"):
        super().__init__(type)
        self._type = type

    @debuggable
    def __call__(
        self, tokens: List[Token], ast: Tree
    ) -> Iterator[Tuple[List[LabeledToken], List[LabeledEdge], SentenceInfo]]:
        """identity transformation with additional data flow graph info parsed (java)

        Returns:
            Iterator[Tuple[List[LabeledToken], SentenceInfo]]: code and SentenceInfo containing List of Tuple[int, int, Labels] - dataflow nodes and random spurious nodes
            Labels:
                0 -- no edge
                1 -- comesFrom
                2 -- computedFrom
        """

        dfg = get_data_flow(tokens, ast)
        # candidates = []
        # for idx, token in enumerate(tokens):
        #     if token.node.type in set(
        #         {'identifier': 152019,
        #         'type_identifier': 4647,
        #         'null_literal': 591,
        #         'string_literal': 1637,
        #         'decimal_integer_literal': 2042,
        #         'hex_integer_literal': 38,
        #         'void_type': 64,
        #         'character_literal': 33,
        #         'decimal_floating_point_literal': 93,
        #         'boolean_type': 19,
        #         }.keys()
        #     ):
        #         candidates.append(idx)
        def map_label(x):
            if x[-1] == "comesFrom":
                return LabeledEdge(first=x[0], second=x[1], label=Labels.comesFrom)
            elif x[-1] == "computedFrom":
                return LabeledEdge(first=x[0], second=x[1], label=Labels.computedFrom)
            else:
                raise ValueError("unexpected edge type")

        positive_edges = [
            map_label(dfg_edge) for dfg_edge in dfg
        ]  # list(int, int, label)

        # now generate negative edges
        used_edges = set((edge.first, edge.second) for edge in positive_edges)
        used_nodes = set()
        for edge in positive_edges:
            used_nodes.add(edge.first)
            used_nodes.add(edge.second)

        all_edges = [(i, j) for i in used_nodes for j in used_nodes if i != j]
        print(len(all_edges))
        filtered_edges = list(filter(lambda edge: not edge in used_edges, all_edges))
        negative_edges = list(
            filter(lambda edge: not edge in used_edges, filtered_edges)
        )

        idx = np.random.choice(
            np.arange(len(negative_edges)),
            size=min(len(negative_edges), len(positive_edges)),
            replace=False,
        )
        selected_negative_edges = [negative_edges[i] for i in idx]
        selected_negative_edges = [
            LabeledEdge(edge[0], edge[1], Labels.NoEdge)
            for edge in selected_negative_edges
        ]

        labeled_edges = positive_edges + selected_negative_edges

        yield [
            LabeledToken(token.value, None) for token in tokens
        ], labeled_edges, SentenceInfo(None)


if __name__ == "__main__":
    aug = IdentityWithDataFlowGraph()
    code = """
    public static void printres ( ) {
    int x1 = 1 ;
    int y1 = 0 ;
    if ( x1 == 0 ) {
        y2 = x1 + y1;
    } else {
        x3 = y1 ;
    }
    }
    """.replace(
        "\n", " "
    )
    while "  " in code:
        code = code.replace("  ", " ").strip()
    print(code)
    tokens, ast = aug.process(code)
    for _, sentencelabel in aug(tokens, ast):
        for idx, from_idx, edge_type in sentencelabel.label:
            print(
                (tokens[idx].value, idx), (tokens[from_idx].value, from_idx), edge_type
            )
    """ 
    ('x1', 8) ('1', 10) comesFrom
    ('y1', 13) ('0', 15) comesFrom
    ('x1', 19) ('x1', 8) comesFrom
    ('y2', 24) ('x1', 26) computedFrom
    ('x1', 26) ('x1', 8) comesFrom
    ('x3', 32) ('y1', 34) computedFrom
    ('y1', 34) ('y1', 13) comesFrom
    """
