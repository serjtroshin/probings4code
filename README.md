# Probing Pretrained Models of Source Code
This project contains source code to replicate the experiments from the [Probing Pretrained Models of Source Code](https://arxiv.org/abs/2202.08975) accepted to [EMNLP 2022 Black Box NLP Workshop](https://blackboxnlp.github.io).

## Overview
The code structure is as follows:
- scripts for running experiments
- `src`:
  - `models`, pretrained models
  - `struct_probing`:
    - `code_augs` syntetic changes to code
    - `probings` utils for probings
- `CodeAnalysis`: directory with processed data and results

## Repository Environment
- `git clone https://github.com/serjtroshin/probings4code`
- `git lfs fetch --all`
  
The script was run on `CentOS Linux 7`, `Python 3.9.2`.
Create a conda environment for the project and install requirements:
- `conda create -n probings4code python=3.9.12`
- `conda activate probings4code`
- `conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.6 numpy=1.22.3 -c pytorch -y`
- `pip install -r requirements.txt`

Install tree-sitter parser for `python` and `java` by running 
-  `bash build.sh`

## Data preprocessing
To prepare data for the tasks run and create `train`, `test` splits: 
- `bash generate_synt_data.sh`

The script will output `all.json` file with train/test splits `train.json`, `test.json` in the following subfolders in CodeAnalysis directory: `identity`,`undeclared`, `dfg`, `identname`,  `varmisuse`, `readability`, `algo`.

To prepare data for ablation study (Appendix) run:
- `bash prepare_ablation.sh`

## Downloading models
`src/models` directory contains a folder for each pretrained model. 

To download BERT, CodeBERT, GraphCodeBERT, CodeT5, and GPT HugginFace checkpoints and tokenizers use:
  - `bash download_models.sh`

To run the experiments with PLBART models, please download PLBART pretrained `plbart_base`, `plbart_large` checkpoints from the original [PLBART official repository](https://github.com/wasiahmad/PLBART) putting them in the `src/models/plbart` folder. Finetuned checkpoints are also avaliable in the official PLBART repo to reproduce the experiments comparing finetuned models (Figure 5). Use `src/models/available_models.py` to provide relevant paths for checkpoints.

## Saving embedding to disk
To save embeddings from all layers for all tasks use 
- `bash save_embeddings.sh`. 

The script will save embeddings to the `data_all.pkz` in the `CodeAnalysis` subfolders.

- NOTE: saving all embeddings requires up to 10TB of disk space.

### Running probing experiments
The run_parallel.py script runs the probing experiments for all models for all tasks saving the results in `csv` format for each model-probing pair at `CodeAnalysis` directory.

To replicate the experiments with linear probing model (Figure 3, 4) use:
- `python3 run_parallel.py` to run experiments with the *linear model*

To run the probing experiments with a *3-layer MLP*:
`python3 run_parallel.py --probing_model mlp`

To run the experiments for ablation study (Appendix) use:
- `bash run_ablation.sh`

Note you can pass `--model <model_name> --probing <probing_task_name>` flags to `run_parallel.py` to run the particular model on the particular task.

# Acknowledgements
We use the following projects in our work:
- [Fairseq](https://github.com/pytorch/fairseq)
- [PLBART](https://github.com/wasiahmad/PLBART)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Hugging Face](https://pypi.org/project/transformers/)

# Citation
If you found this code useful, please cite our work:
```
@misc{https://doi.org/10.48550/arxiv.2202.08975,
    doi = {10.48550/ARXIV.2202.08975},
    url = {https://arxiv.org/abs/2202.08975},
    author = {Troshin, Sergey and Chirkova, Nadezhda},
    keywords = {Software Engineering (cs.SE), Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Probing Pretrained Models of Source Code},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```
