import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import jsonlines

tokenizer = AutoTokenizer.from_pretrained("gpt2")

path = '/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/mauve-human-eval-anon.csv'
df = pd.read_csv(path)

print(df.shape)
index_list = []
truncate_a = []
truncate_b = []
for index, row in df.iterrows():
    if row['Input.model_b'] == 'human' or row['Input.model_a'] == 'human':
        index_list.append(index)

        encoded_input_a = tokenizer(row['Input.completiona'],
                                    max_length=256,
                                    add_special_tokens=False)
        encoded_input_b = tokenizer(row['Input.completionb'],
                                    max_length=256,
                                    add_special_tokens=False)
        truncate_a.append(row['Input.ctx'] +
                          tokenizer.decode(encoded_input_a['input_ids']))
        truncate_b.append(row['Input.ctx'] +
                          tokenizer.decode(encoded_input_b['input_ids']))

        # print(row['Input.completiona'])
        # print(row['Input.completionb'])

targt_df = df.iloc[index_list]

targt_df.to_csv('human_mauve.csv', index=False)

targt_df['trun_a'] = truncate_a
# targt_df.to_json(
#     '/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/tunc_a.jsonl',
#     lines=True,
#     orient='records')

targt_df['trun_b'] = truncate_b
# targt_df.to_json(
#     '/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/trun_b.jsonl',
#     lines=True,
#     orient='records')

sort_rows = []
source = [
    "('gpt2', 'p0.9')", "('gpt2', 'p1.0')", "('gpt2-large', 'p0.95')",
    "('gpt2-large', 'p1.0')", "('gpt2-medium', 'p0.9')",
    "('gpt2-medium', 'p1.0')", "('gpt2-xl', 'p0.95')", "('gpt2-xl', 'p1.0')"
]
for s in source:
    for index, row in targt_df.iterrows():
        if row['Input.model_b'] == s or row['Input.model_a'] == s:
            with jsonlines.open(
                    f'/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/{s}_a.jsonl',
                    'a') as f:
                row['text'] = row['Input.ctx'] + ' '+ row['Input.completiona']
                f.write(row.to_dict())
            with jsonlines.open(
                    f'/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/{s}_b.jsonl',
                    'a') as f:
                row['text'] = row['Input.ctx'] + ' '+ row['Input.completionb']
                f.write(row.to_dict())
# sorted_df = pd.DataFrame(sort_rows)
# sorted_df.to_csv('/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/sorted.csv', index=False)

# print('The number of pieces of data:', targt_df.shape)
# targt_df.to_csv(
#     '/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/mauve-human-eval-anon-human_clean.csv',
#     index=False)
