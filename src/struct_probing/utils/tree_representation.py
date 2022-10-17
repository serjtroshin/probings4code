from __future__ import annotations

from typing import Iterator, List

from src.struct_probing.utils.code_parser import Tree


class NodeRepresentation:
    """AST node with its representation"""

    def __init__(self, node: Tree, representation: List[int]):
        self.node = node
        self.representation = representation

    def __str__(self) -> str:
        return f"{str(self.node)}, {str(self.representation)}"


def is_identifier(node) -> bool:
    return node.type == "identifier"


def get_node_representations(tree: Tree) -> Iterator[NodeRepresentation]:
    """extract node representations from AST tree. Node representation
    is encoded as a path from the root, by enumerating the children.

    Args:
        tree (Tree): AST

    Yields:
        Iterator[NodeRepresentation]: representations
    """
    representation: List[int] = []

    cursor = tree.walk()

    reached_root = False
    while not reached_root:

        yield NodeRepresentation(cursor.node, representation[:])

        if cursor.goto_first_child():
            representation.append(0)
            continue

        if cursor.goto_next_sibling():
            representation[-1] += 1
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            else:
                representation.pop()

            if cursor.goto_next_sibling():
                representation[-1] += 1
                retracing = False


class NodeRepr:
    def __init__(self, vec: List[int]):
        self.vec = vec

    def __str__(self) -> str:
        return f"vec:{str(self.vec)}"

    def dist(self, other: NodeRepr) -> int:
        """calculates distance in a tree between two nodes

        Args:
            other ([type]): NodeRepr

        Returns:
            [type]: [description]
        """
        i = 0
        while i < len(self.vec) and i < len(other.vec) and self.vec[i] == other.vec[i]:
            i += 1
        return (len(self.vec) - i) + (len(other.vec) - i)


if __name__ == "__main__":
    import numpy as np
    from code_parser import CodeParser

    sent = "public static void setState ( State state ) { currentState = state ; }"
    code = CodeParser("java")(sent)
    nodes = get_node_representations(code.tree)
    nvs = list(filter(lambda p: is_identifier(p.node), nodes))
    nvs_repr = list(NodeRepr(p.representation) for p in nvs)
    for elem in nvs_repr[:3]:
        print(elem)

    dist = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dist[i, j] = nvs_repr[i].dist(nvs_repr[j])
    print(dist)
