from io import BytesIO
import json
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP, ENCODING
from keyword import iskeyword
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import shutil
import logging 
import argparse

log = logging.getLogger("code_augs")
log.setLevel(logging.INFO)

def print_tok(tok):
    print(f"Type: {tok.type}\nString: {tok.string}\nStart: {tok.start}\nEnd: {tok.end}\nLine: {tok.line.strip()}\n======\n")

def tokens(code):
    for tok in tokenize(BytesIO(code.encode('utf-8')).read):
        if tok.type != ENCODING:
            yield tok

    
def sub_tokens(code, all_tokens, tokens, value=lambda v: "MASK"):
    def get_pos(tok):
        assert tok.start[0] == tok.end[0]
        return (tok.start[0], (tok.start[1], tok.end[1]))
    if len(tokens) == 0 or len(all_tokens) == 0:
        return code
    
    new_code = []
    pos = 0
    for tok in tokens:
        new_code.append(code[pos: tok.start[1]])
        new_code.append(value(tok.string))
        pos = tok.end[1]
        
    new_code.append(code[pos:])
    return "".join(new_code)
    
def get_keywords(code):
    for tok in tokens(code):
        if iskeyword(tok.string):
            yield tok

def get_identifiers(code):
    for tok in tokens(code):
        if tok.type in {NAME, STRING} and not iskeyword(tok.string):
            yield tok
            
class anonimizer:
    def __init__(self):
        self.d = dict()
        self.cnt = 1
    def __call__(self, v):
        if v in self.d:
            return self.d[v]
        else:
            self.d[v] = f"var{self.cnt}"
            self.cnt += 1
            return self.d[v]
        
def get_punkt(code):
    PUNKT = set("""
    +       -       *       **      /       //      %      @
    <<      >>      &       |       ^       ~       :=
    <       >       <=      >=      ==      !=
    (       )       [       ]       {       }
    ,       :       .       ;       @       =       ->
    +=      -=      *=      /=      //=     %=      @=
    &=      |=      ^=      >>=     <<=     **=
    """.strip().split())
    for tok in tokens(code):
        if not tok.type in {NAME, STRING} and not iskeyword(tok.string) and tok.string in PUNKT:
            yield tok

def get_code_augs(code):
    all_tokens = [tok for tok in tokens(code)]

    yield ("default", code)
    
    keywords = [tok for tok in get_keywords(code)]
    yield ("keyword", sub_tokens(code, all_tokens, keywords))
    
    identifiers = [tok for tok in get_identifiers(code)]
    yield ("ident", sub_tokens(code, all_tokens, identifiers, value=anonimizer()))
    
    
    punkt = [tok for tok in get_punkt(code)]
    yield ("punkt", sub_tokens(code, all_tokens, punkt))
    
        
def main(args):
    data = None
    with open(f"{args.data_dir}/{args.task}/all.json", "r") as f:
        data = json.load(f)
    
    data_aug = defaultdict(list)
    for elem in data:
        try:
            for name, new_code in get_code_augs(elem["code_joined"].strip()):
                elem_copy = deepcopy(elem)
                elem_copy["code"] = new_code
                elem_copy["code_joined"] = new_code
                data_aug[name].append(elem_copy)
        except Exception as e:
            print(e)

    for key in data_aug:
        save_to = f"{args.output_dir}/{args.task}__{key}/"
        Path(save_to).mkdir(parents=True, exist_ok=True)
        with open(save_to + "all.json", "w") as f:
            json.dump(data_aug[key], f)
        for mode in "train", "test":
            shutil.copy(f"{args.data_dir}/{args.task}/{mode}.json", save_to + f"{mode}.json")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="CodeAnalysis", help="data-dir")
    parser.add_argument("--output_dir", default="CodeAnalysisAug", help="output-dir")

    parser.add_argument("--task", default="algo_1671-A/algo", help="data-dir")
    parser.add_argument("--n_samples", type=int, default=1000000)

    # Debug
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    main(args)
