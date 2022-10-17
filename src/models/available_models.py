from src.models.bert.model import BERT
from src.models.codebert.model import CodeBertModel
from src.models.codet5.model import CodeT5Model, CodeT5ModelSmall
from src.models.gpt.model import CodeGPT2Model
from src.models.graphcodebert.model import GraphCodeBertModel
from src.models.plbart.model import PLBARTModel, PLBARTModelLarge

MODELS = {
    "CodeT5": CodeT5Model,
    "CodeT5_small": CodeT5ModelSmall,
    "MLM": PLBARTModel,
    "plbart_csn": PLBARTModel,
    "plbart_large": PLBARTModelLarge,
    "CodeBert": CodeBertModel,
    "GraphCodeBert": GraphCodeBertModel,
    "CodeGPT2Model": CodeGPT2Model,
    "BERT": BERT,
}

# src/models/plbart/<checkpoint_path>/checkpoint_best.pt
finetuned_paths = {
    "finetuned_from_MLM_clone_detection", "finetuned_from_MLM_code_generation","finetuned_from_MLM_code_summarization", "finetuned_from_MLM_code_translation", "finetuned_from_MLM_defect_prediction", "finetuned_no_pretrain_clone_detection","finetuned_no_pretrain_code_generation", "finetuned_no_pretrain_code_summarization", "finetuned_no_pretrain_code_translation", "finetuned_no_pretrain_defect_prediction"
}
FINETUNED_MODELS = {}
for checkpoint_path in finetuned_paths:
    FINETUNED_MODELS[checkpoint_path] = (PLBARTModel, checkpoint_path)
