# from codeaug.varm import (Language, Node, Parser, TreeSitterCode,
#                           tree_sitter_path)

from pathlib import Path
from typing import Generator
from tree_sitter import Language, Parser, Tree, Node


def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


class TreeSitterCode:
    """
    stores 1) the source code in bytes
           2) the tree
        allows to traverse nodes in the AST tree, replace values.
        updates both with `replace` method
        (tool to change identifiers)
    """
    def __init__(self, code: str, parser: Parser):
        self.code_bytes = bytes(
            code, "utf-8"
        )
        self.parser = parser
        self.tree = self.parser.parse(self.code_bytes)
        
    @property
    def code(self):
        return self.code_bytes.decode()
    
    def traverse(self) -> Generator[Node, None, None]:
        for item in traverse_tree(self.tree):
            yield item
            
    def identifiers(self):
        def is_identifier(node): return node.type=='identifier'
        for node in self.traverse():
            if is_identifier(node):
                yield node
                
    def get_value(self, node):
        return self.code_bytes[node.start_byte: node.end_byte]
            
DIR = Path(__file__).parent.absolute()
tree_sitter_path=str(Path(DIR, "parser", "my-languages.so"))

class CodeParser:
    def __init__(self, lang: str):
        assert lang in ["python", "java"]
        LANGUAGE = Language(tree_sitter_path, lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        self.parser = parser

    def __call__(self, code_example, p=0.0) -> TreeSitterCode:
        code = TreeSitterCode(code_example, self.parser)
        return code
