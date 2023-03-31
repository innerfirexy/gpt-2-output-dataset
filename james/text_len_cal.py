import pandas as pd
from transformers import AutoTokenizer


path = "/Users/james/Workspace/gpt-2-output-dataset/data/"
with open(path + "webtext.test.jsonl") as f:
    df = pd.read_json(f, lines=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

result = [0] * 5
for index, row in df.iterrows():
    # print(text))
    # print(len(text)))
    text = row["text"]
    text = tokenizer(text, truncation=True, max_length=1024)["input_ids"]
    # print(text)
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
# [3554, 19, 1, 1, 1697] 5272 GLM10B
# [1002, 944, 751, 526, 1777] 5000 Human