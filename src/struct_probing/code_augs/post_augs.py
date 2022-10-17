from regex import E

from .aug import CodeAugmentation, LabeledTokenTyped


class JavaToken(object):
    def __init__(self, value, position=None, javadoc=None):
        self.value = value
        self.position = position
        self.javadoc = javadoc

    def __repr__(self):
        if self.position:
            return '%s "%s" line %d, position %d' % (
                self.__class__.__name__,
                self.value,
                self.position[0],
                self.position[1],
            )
        else:
            return '%s "%s"' % (self.__class__.__name__, self.value)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        raise Exception("Direct comparison not allowed")


class EndOfInput(JavaToken):
    pass


class Keyword(JavaToken):
    VALUES = set(
        [
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "goto",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
        ]
    )


class Modifier(Keyword):
    VALUES = set(
        [
            "abstract",
            "default",
            "final",
            "native",
            "private",
            "protected",
            "public",
            "static",
            "strictfp",
            "synchronized",
            "transient",
            "volatile",
        ]
    )


class BasicType(Keyword):
    VALUES = set(["boolean", "byte", "char", "double", "float", "int", "long", "short"])


class Literal(JavaToken):
    pass


class Integer(Literal):
    pass


class DecimalInteger(Literal):
    pass


class OctalInteger(Integer):
    pass


class BinaryInteger(Integer):
    pass


class HexInteger(Integer):
    pass


class FloatingPoint(Literal):
    pass


class DecimalFloatingPoint(FloatingPoint):
    pass


class HexFloatingPoint(FloatingPoint):
    pass


class Boolean(Literal):
    VALUES = set(["true", "false"])


class Character(Literal):
    pass


class String(Literal):
    pass


class Null(Literal):
    pass


class Separator(JavaToken):
    VALUES = set(["(", ")", "{", "}", "[", "]", ";", ",", "."])


class Operator(JavaToken):
    MAX_LEN = 4
    VALUES = set(
        [
            ">>>=",
            ">>=",
            "<<=",
            "%=",
            "^=",
            "|=",
            "&=",
            "/=",
            "*=",
            "-=",
            "+=",
            "<<",
            "--",
            "++",
            "||",
            "&&",
            "!=",
            ">=",
            "<=",
            "==",
            "%",
            "^",
            "|",
            "&",
            "/",
            "*",
            "-",
            "+",
            ":",
            "?",
            "~",
            "!",
            "<",
            ">",
            "=",
            "...",
            "->",
            "::",
        ]
    )

    # '>>>' and '>>' are excluded so that >> becomes two tokens and >>> becomes
    # three. This is done because we can not distinguish the operators >> and
    # >>> from the closing of multipel type parameter/argument lists when
    # lexing. The job of potentially recombining these symbols is left to the
    # parser

    INFIX = set(
        [
            "||",
            "&&",
            "|",
            "^",
            "&",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "<<",
            ">>",
            ">>>",
            "+",
            "-",
            "*",
            "/",
            "%",
        ]
    )

    PREFIX = set(["++", "--", "!", "~", "+", "-"])

    POSTFIX = set(["++", "--"])

    ASSIGNMENT = set(
        ["=", "+=", "-=", "*=", "/=", "&=", "|=", "^=", "%=", "<<=", ">>=", ">>>="]
    )

    LAMBDA = set(["->"])

    METHOD_REFERENCE = set(
        [
            "::",
        ]
    )

    def is_infix(self):
        return self.value in self.INFIX

    def is_prefix(self):
        return self.value in self.PREFIX

    def is_postfix(self):
        return self.value in self.POSTFIX

    def is_assignment(self):
        return self.value in self.ASSIGNMENT


class Annotation(JavaToken):
    pass


class Identifier(JavaToken):
    pass


class Comment(JavaToken):
    pass


MASK = "MASK"


class MaskAugmentation:
    def __init__(self, name="default"):
        self.name = name

    def mask(self, token: LabeledTokenTyped):
        return False

    def postprocess(self, token: LabeledTokenTyped):
        return token

    def __call__(self, token: LabeledTokenTyped):
        if self.mask(token):
            ret = LabeledTokenTyped(value=MASK, type=token.type, label=token.label)
        else:
            ret = token
        ret = self.postprocess(ret)
        return ret


class MaskPunkt(MaskAugmentation):
    def __init__(self, name="punkt"):
        super().__init__(name)
        self.punkt_set = (
            set()
            .union(Separator.VALUES)
            .union(Operator.VALUES)
            .union(Operator.INFIX)
            .union(Operator.PREFIX)
            .union(Operator.ASSIGNMENT)
        )
        # print("punktuation", self.punkt_set)

    def mask(self, token: LabeledTokenTyped):
        return token.value in self.punkt_set


class MaskIdentifiers(MaskAugmentation):
    def __init__(self, name="ident"):
        super().__init__(name)
        # self.identifier = {"identifier"}
        # print(self.name, self.identifier)
        self.cnt = 1
        self.idents = dict()

        # self.identifier = {
        #     'identifier', 'string_literal', 'character_literal', "scoped_identifier",
        # }
        self.punkt_set = (
            set()
            .union(Separator.VALUES)
            .union(Operator.VALUES)
            .union(Operator.INFIX)
            .union(Operator.PREFIX)
            .union(Operator.ASSIGNMENT)
        )
        self.keywrd = (
            set().union(Keyword.VALUES).union(Modifier.VALUES).union(BasicType.VALUES)
        )

    def postprocess(self, token: LabeledTokenTyped):
        if (
            token.value != "var"
            and not token.value in self.keywrd
            and not token.value in self.punkt_set
        ):
            if not token.value in self.idents:
                self.idents[token.value] = f"var{self.cnt}"
                self.cnt += 1
            return LabeledTokenTyped(
                value=self.idents[token.value], type=token.type, label=token.label
            )
        else:
            return token

    def mask(self, token: LabeledTokenTyped):
        return False  # var is special for augs


class MaskKeywords(MaskAugmentation):
    def __init__(self, name="keyword"):
        super().__init__(name)
        self.keywrd = (
            set().union(Keyword.VALUES).union(Modifier.VALUES).union(BasicType.VALUES)
        )
        # print(self.name, self.keywrd)

    def mask(self, token: LabeledTokenTyped):
        return token.value in self.keywrd