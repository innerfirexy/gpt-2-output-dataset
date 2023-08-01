from transformers import AutoTokenizer

# GPT2-sm
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokens = tokenizer("Hello world")["input_ids"]
print(tokens)