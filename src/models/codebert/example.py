import torch
from transformers import AutoModel, AutoTokenizer

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    tokenizer.save_pretrained("./codebert-base")
    model.save_pretrained("./codebert-base")

    tokenizer = AutoTokenizer.from_pretrained("./codebert-base")
    model = AutoModel.from_pretrained("./codebert-base")
    model = model.to(device)

    nl_tokens = tokenizer.tokenize("return maximum value")
    code_tokens = tokenizer.tokenize("def max ( a , b ): if a > b: return a else return b")
    tokens = (
        [tokenizer.cls_token]
        + nl_tokens
        + [tokenizer.sep_token]
        + code_tokens
        + [tokenizer.sep_token]
    )
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("tokens_ids", tokens_ids)

if __name__ == "__main__":
    run()