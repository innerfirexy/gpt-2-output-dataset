import pandas as pd
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm

path = "/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/5273_sample/"
gen_path = "/Users/james/Workspace/gpt-2-output-dataset/james/split/"
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(path + "webtext.train.model.sorted.jsonl") as f:
    df = pd.read_json(f, lines=True)

var_len = 0
for index, row in tqdm(df.iterrows()):
    if row['token_len'] >= var_len and row["token_len"] < var_len + 200:
        with jsonlines.open(
                gen_path + "webtext.train.split." + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'prompt': row['prompt'], 'text': row["text"]})
    else:
        var_len += 200
        with jsonlines.open(
                gen_path + "webtext.train.split." + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'prompt': row['prompt'], 'text': row["text"]})

