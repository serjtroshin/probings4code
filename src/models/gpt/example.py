from transformers import GPT2Model, GPT2Tokenizer

model_name = "microsoft/CodeGPT-small-java-adaptedGPT2"
# model_name = "./CodeGPT-small-java-adaptedGPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # CodeGPT-small-java-adaptedGPT2
model = GPT2Model.from_pretrained(model_name)

tokenizer.save_pretrained(f"./{model_name}")
model.save_pretrained(f"./{model_name}")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
# print(model)
output = model(**encoded_input, output_hidden_states=True)
# print(len(output["hidden_states"]))
# print(output["hidden_states"][0].shape)
