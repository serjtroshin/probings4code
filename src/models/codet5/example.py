from transformers import (RobertaTokenizer, T5ForConditionalGeneration)

def run():
    for model_name in "codet5-small", "codet5-base":
        tokenizer = RobertaTokenizer.from_pretrained(f"Salesforce/{model_name}")
        model = T5ForConditionalGeneration.from_pretrained(f"Salesforce/{model_name}")
        tokenizer.save_pretrained(f"./tokenizer-{model_name}")
        model.save_pretrained(f"./model-{model_name}")

        text = "def greet(user): print(f'hello <extra_id_0> <extra_id_1>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        # simply generate one code span
        generated_ids = model.generate(input_ids, max_length=8)
        # print(generated_ids)
        # print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        # this prints "{user.username}"


if __name__ == "__main__":
    run()