for model in "MLM" "BERT" "CodeT5" "CodeGPT2Model" "CodeT5_small" "CodeBert" "GraphCodeBert"  "plbart_large" "finetuned_from_MLM_clone_detection" "finetuned_from_MLM_code_generation" "finetuned_from_MLM_code_summarization" "finetuned_from_MLM_code_translation" "finetuned_from_MLM_defect_prediction" "finetuned_no_pretrain_clone_detection" "finetuned_no_pretrain_code_generation" "finetuned_no_pretrain_code_summarization" "finetuned_no_pretrain_code_translation" "finetuned_no_pretrain_defect_prediction" 
do
    for aug in algo dfg varmisuse readability identname undeclared identity;
    do
        python3 save_embeddings.py --model ${model} --insert ${aug} --n_samples 10000
    done
done
python3 get_all_invalid_ids.py  # to filter out failed code snippets