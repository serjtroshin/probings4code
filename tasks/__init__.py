from tasks.funcname.dataset import \
    get_dataset as funcname_get_dataset
from tasks.mlm.dataset import get_dataset as mlm_get_dataset
from tasks.readability.dataset import \
    get_dataset as readability_get_dataset
from tasks.sorts.dataset import get_dataset as sorts_get_dataset

DATASETS = {
    "mlm": mlm_get_dataset,
    "readability": readability_get_dataset,
    "funcname": funcname_get_dataset,
    "sorts": sorts_get_dataset,
}
