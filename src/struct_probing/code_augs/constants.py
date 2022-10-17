class TokenTypes:
    Identifier = "identifier"


FILL_VALUE = {
    "float": "0.0f",
    "int": "0",
    "double": "0.0d",
    "long": "0L",
    "char": "\0",
    "short": "0",
    "byte": "0",
    "boolean": "false",
}


class NodeTypes:
    AssignmentExpression = "assignment_expression"
    Block = "block"
    ExpressionStatement = "expression_statement"
    ForStatement = "for_statement"
    Identifier = "identifier"
    IntegralType = "integral_type"
    FloatingPointType = "floating_point_type"
    LocalVariableDeclaration = "local_variable_declaration"
    VariableDeclarator = "variable_declarator"


class RequiredEmbeddings:
    MEAN = "mean"
    DUMMY = "dummy"  # identity
