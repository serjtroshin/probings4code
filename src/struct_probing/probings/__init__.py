from typing import Dict, List

from src.struct_probing.probings import (clone_detection, edge, sentence_classification,
                      sentence_regression, token_classification,
                      token_regression)
from src.struct_probing.probings.base import ProbingTask
from src.struct_probing.probings.models import ProbingModelType

_supported_probings: List[ProbingTask] = []
_supported_probings.extend(sentence_regression.PROBINGS)
_supported_probings.extend(sentence_classification.PROBINGS)
_supported_probings.extend(token_classification.PROBINGS)
_supported_probings.extend(token_regression.PROBINGS)
_supported_probings.extend(edge.PROBINGS)
# _supported_probings.extend(clone_detection.PROBINGS)

supported_probings: Dict[str, ProbingTask] = {}
for probing in _supported_probings:
    supported_probings[probing.get_name()] = probing


# supported_dataset_specific_probings = {
#     probing.name: probing
#     for probing in sentence_classification.DATASET_SPECIFIC_PROBINGS
# }
supported_dataset_specific_probings = {}
