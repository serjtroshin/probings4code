import torch
from transformers import AutoModel, AutoTokenizer


def run():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    tokenizer.save_pretrained("./graphcodebert-base")
    model.save_pretrained("./graphcodebert-base")

    tokenizer = AutoTokenizer.from_pretrained("./graphcodebert-base")
    model = AutoModel.from_pretrained("./graphcodebert-base")
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
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]

    with torch.no_grad():
        generated_ids = model.forward(
            torch.tensor(tokens_ids)[None, :], output_hidden_states=True
        )
        # print(len(generated_ids.hidden_states))

if __name__ == "__main__":
    run()
