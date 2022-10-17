set -e
for model in BERT MLM; do
    for ablation in "default" "ident" "keyword" "punct";
    do
        python3 run_parallel.py --post_aug $ablation --model $model
    done
done