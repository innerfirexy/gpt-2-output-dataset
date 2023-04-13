import pandas as pd
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm

path = "/root/autodl-tmp/gpt-2-output-dataset/data/"
gen_path = "/root/autodl-tmp/gpt-2-output-dataset/data/"
# postfix = 'valid' # train_old /valid /test
postfix = 'train_opt_125m_top_50_wiki' # train_opt_size_sampling_method_domain
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(path + f"webtext.{postfix}.sorted.jsonl") as f:
    df = pd.read_json(f, lines=True)

var_len = 0
gap = 200
for index, row in tqdm(df.iterrows()):
    if row['token_len'] >= var_len and row["token_len"] < var_len + gap:
        with jsonlines.open(
                gen_path + f"webtext.{postfix}.split_" + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'text': row["text"]})
        with open(
                gen_path + f"webtext.{postfix}.split_" + str(var_len) + ".nll",
                "a") as we:
            entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
            we.write(f'{entropy}\n')
    else:
        var_len += 200
        if var_len + gap == 1000:
            gap = 225
        with jsonlines.open(
                gen_path + f"webtext.{postfix}.split_" + str(var_len) + ".jsonl",
                "a") as w:
            w.write({'text': row["text"]})
        with open(
                gen_path + f"webtext.{postfix}.split_" + str(var_len) + ".nll",
                "a") as we:
            entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
            we.write(f'{entropy}\n')
            