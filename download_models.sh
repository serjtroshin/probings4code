cur_dir=$(pwd)
for model_name in "bert" "codebert" "graphcodebert" "codet5" "gpt" ;
do
    cd $cur_dir
    cd src/models/${model_name}
    python3 example.py
done