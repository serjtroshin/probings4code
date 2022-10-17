# ablation study
for MODEL in "MLM" "BERT" 
do
    for aug in dfg undeclared algo identity varmisuse readability identname 
    do
        for post_aug in default ident keyword punkt
        do
            python3 save_embeddings_post_aug.py --model ${MODEL} --insert ${aug} --post_aug_name ${post_aug}
        done
    done
done