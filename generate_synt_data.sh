for task in identity, undeclared, dfg, identname, readability, varmisuse, algo;
do
    python3 generate_synt_data.py --task ${task} --n_samples 100000
done