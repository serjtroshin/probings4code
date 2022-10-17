from transformers import BertModel, BertTokenizer

# https://huggingface.co/bert-base-cased

def run():
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    tokenizer.save_pretrained(f"./{model_name}")
    model.save_pretrained(f"./{model_name}")


    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input, output_hidden_states=True)

    # print(len(output["hidden_states"]))
    # print(output["hidden_states"][0].shape)
    
if __name__ == "__main__":
    run()
