import json
import nltk
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_name = "opt-6.7b"
model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}", torch_dtype=torch.float16).cuda()

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}", use_fast=False) # must be the one associated with the pretrained model

# get the prompt_text
prompt_list = []
opt_list = []
print("Reading prompt_text......")
with open("story_vary.txt", "r") as file:
    content_lines = file.readlines()
prompt_list = [content_lines[i].rstrip('\n') for i in range(1, 15000, 3)]
for idx, prompt in tqdm(enumerate(prompt_list)):
    opt_list.append({"id": idx, "ended": False, "length": 0, "text": ""})

# generate text using OPT models
print("Generating opt_text......")
for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
    input_ids = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").input_ids.cuda()
    set_seed(32) # random seed
    generated_ids = model.generate(input_ids, do_sample=True, max_length=1024) # generated by top-k(default: k=50) sampling
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    length = len(nltk.word_tokenize(gen_text))
    if gen_text[-1] == ".":
        opt_list[idx]["ended"] = True
    opt_list[idx]["length"] = length
    opt_list[idx]["text"] = gen_text

# output generated text
print("Wrting opt_text......")
with open("webtext.train_opt_6.7b_top_50_story.jsonl", "w") as file:
    for line in tqdm(opt_list, total=len(opt_list)):
        json.dump(line, file)
        file.write("\n")
