set -e
for aug in varmisuse identity identname unidentified_var_hard_fixed dfg readability
do
    python3 post_augs_java.py --insert_bug $aug
done
python3 post_augs_python.py