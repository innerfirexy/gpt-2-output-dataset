import json
import nltk
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "gpt2-xl"], help="Model name to use (default: gpt2).")
parser.add_argument("--prompt_text", type=str, default="story_vary", choices=["story_vary", "truenews_35", "wikitext_35"], help="Prompt text for generation (default: story_vary).")
args = parser.parse_args()

model_name = args.model_name
prompt_text = args.prompt_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# get the prompt_text
prompt_list = []
opt_list = []
print("Reading prompt_text......")
with open(f"../zuhao/prompt_raw/{prompt_text}.txt", "r", encoding='utf-8') as file:
    content_lines = file.readlines()
MAXLINES = 5000
prompt_list = [content_lines[i].rstrip('\n') for i in range(1, 3*MAXLINES, 3)]
for idx, prompt in tqdm(enumerate(prompt_list)):
    opt_list.append({"id": idx, "ended": False, "length": 0, "gen_text": ""})

# generate text using GPT2 models
print("Generating text......")
for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
    set_seed(32)
    encodings = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).float().to(device)
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_length=1024) # generated by top-k sampling
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    length = len(nltk.word_tokenize(gen_text))
    if gen_text[-1] == ".":
        opt_list[idx]["ended"] = True
    opt_list[idx]["length"] = length
    opt_list[idx]["gen_text"] = gen_text
    # # just for test
    # print(f"prompt: {prompt}")
    # print(f"gen_text: {gen_text}")

# output generated text
print("Wrting gpt2_text......")
with open(f"../data/gpt2-generated-from-prompt/{model_name}-{prompt_text}.jsonl", "w") as file:
    for line in tqdm(opt_list, total=len(opt_list)):
        json.dump(line, file)
        file.write("\n")
