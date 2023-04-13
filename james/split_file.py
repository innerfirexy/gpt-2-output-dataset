import pandas as pd
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm

path = "/home/yyuan/gpt-2-output-dataset/data/"
gen_path = "/home/yyuan/gpt-2-output-dataset/data/"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
filename = 'webtext.train.model=.bloom_7b1.wiki.sorted.jsonl'
with open(path + filename) as f:
    df = pd.read_json(f, lines=True)

len_var = 0
len_gap = 200
for i in range(5):
    temp_df = df[(df['token_len'] >= len_var)
                 & (df['token_len'] < len_var + len_gap)]
    temp_df.to_json(gen_path + filename[:-5] + 'split.' + str(len_var) +
                    ".jsonl",
                    orient='records',
                    lines=True)
    for index, row in tqdm(temp_df.iterrows()):
        with open(gen_path + filename[:-5] + 'split.' + str(len_var) + ".nll",
                  "a") as we:
            entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
            we.write(f'{entropy}\n')
    len_var += len_gap
    if len_var == 800:
        len_gap += 25

# var_len = 0
# gap = 200
# for num_rows in split_index:
#     row_count = 0
#     for index, row in tqdm(df.iterrows()):
#         if row_count == num_rows:
#             var_len += gap
#         if row['token_len'] >= var_len and row["token_len"] < var_len + gap:
#             with jsonlines.open(
#                     gen_path + filename[:-5] + 'split.' + str(var_len) + ".jsonl",
#                     "a") as w:
#                 w.write({"prompt": row["prompt"], 'text': row["text"]})
#             with open(gen_path + filename[:-5] + 'split.' + str(var_len) + ".nll",
#                     "a") as we:
#                 # print(row["entropy"], type(row["entropy"]))
#                 entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
#                 we.write(f'{entropy}\n')
#             row_count += 1
