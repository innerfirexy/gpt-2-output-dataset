import pandas as pd
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm

path = "/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/5273_sample/"
gen_path = "/Users/james/Workspace/gpt-2-output-dataset/james/split_human/"
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(path + "webtext.test.model.sorted.jsonl") as f:
    df = pd.read_json(f, lines=True)

var_len = 0
gap = 200
for index, row in tqdm(df.iterrows()):
    if row['token_len'] >= var_len and row["token_len"] < var_len + gap:
        with jsonlines.open(
                gen_path + "webtext.test.split." + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'text': row["text"]})
    else:
        var_len += 200
        if var_len + gap == 1000:
            gap = 225
        with jsonlines.open(
                gen_path + "webtext.test.split." + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'text': row["text"]})

