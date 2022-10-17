from .aug import CodeAugmentation
from .brackets import BracketsCodeAugmentation
# from .variable_insert import SemanticVariableInsert
from .dfg import IdentityWithDataFlowGraph
from .funcname import FuncNamePrediction
from .identname import IdentNamePrediction
from .readability import ReadabilityPrediction
from .sorts import CodeforcesPrediction, SortsPrediction
# from .undeclared import UnidentifiedVariables
from .undeclared_hard import UnidentifiedVariablesHard
from .var_misuse import VarMisuseAugmentation

available_augs = {
    "identity": CodeAugmentation,
    "brackets": BracketsCodeAugmentation,
    # "unidentified_var": UnidentifiedVariables,
    "undeclared": UnidentifiedVariablesHard,
    # "variable_insert": SemanticVariableInsert,
    "dfg": IdentityWithDataFlowGraph,
    "identname": IdentNamePrediction,
    "funcname": FuncNamePrediction,
    "readability": ReadabilityPrediction,
    "sorts": SortsPrediction,
    "varmisuse": VarMisuseAugmentation,
    "algo": CodeforcesPrediction,
}

# post augs
from .post_augs import (MaskAugmentation, MaskIdentifiers, MaskKeywords,
                        MaskPunkt)

post_augs = [MaskAugmentation, MaskIdentifiers, MaskKeywords, MaskPunkt]
