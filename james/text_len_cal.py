import pandas as pd
from transformers import AutoTokenizer

'''

This script is used to calculate the length of the text in the dataset only.

'''

path = "/home/yyuan/bloom_7b1/"
with open(path + "gen_bloom_7b1_story_vary.jsonl") as f:
    df = pd.read_json(f, lines=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

result = [0] * 5
for index, row in df.iterrows():
    text = row["text"]
    text = tokenizer(text, truncation=True, max_length=1024)["input_ids"]
    # print(text, type(text), len(text))
    if len(text) >= 0 and len(text) < 200:
        result[0] += 1
    elif len(text) >= 200 and len(text) < 400:
        result[1] += 1
    elif len(text) >= 400 and len(text) < 600:
        result[2] += 1
    elif len(text) >= 600 and len(text) < 800:
        result[3] += 1
    elif len(text) >= 800 and len(text) <= 1024:
        result[4] += 1

print(result, sum(result))

# [4978, 20, 1, 0, 1] 5000 Bloom_560m News
# [4924, 63, 9, 4, 0] 5000 Bloom_560m Story
# [4928, 61, 8, 1, 2] 5000 Bloom_560m Wiki

# [1608, 688, 410, 271, 2023] 5000 Bloom_7B1 Story

