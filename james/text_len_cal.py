import pandas as pd
from transformers import AutoTokenizer

'''

This script is used to calculate the length of the text in the dataset only.

'''

path = "/Users/james/Workspace/gpt-2-output-dataset/james/bloom_560m/"
with open(path + "gen_bloomz_560m_wikitext_35_cleaned.jsonl") as f:
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

# [4387, 513, 84, 16, 0] 5000 Bloom_560m_Story
# [4431, 448, 86, 26, 9] 5000 Bloom_560m_News
# [3844, 859, 200, 65, 32] 5000 Bloom_560m_TrueNews