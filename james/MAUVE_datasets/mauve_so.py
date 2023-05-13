import pandas as pd
import numpy as np
from transformers import AutoTokenizer
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
        # print(row['Input.ctx'] )
        # print(row['Input.model_a'], '\n')
        # print(row['Input.model_b'], '\n' )
        # print(row['Input.ctx']  + tokenizer.decode(encoded_input_a['input_ids']))
        # row['Input.completiona'] = tokenizer.decode(encoded_input_a['input_ids'])
        truncate_a.append(row['Input.ctx']  + tokenizer.decode(encoded_input_a['input_ids']))
        # row['Input.completionb'] = tokenizer.decode(encoded_input_b['input_ids'])
        truncate_b.append(row['Input.ctx']  + tokenizer.decode(encoded_input_b['input_ids']))

        # print(row['Input.completiona'])
        # print(row['Input.completionb'])

# print(len(index_list))
targt_df = df.iloc[index_list]
targt_df['text'] = truncate_a
targt_df.to_json('/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/tunc_a.jsonl', lines=True, orient='records')

targt_df['text'] = truncate_b
targt_df.to_json('/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/trun_b.jsonl', lines=True, orient='records')

print('The number of pieces of data:', targt_df.shape)
# targt_df.to_csv(
#     '/Users/james/Workspace/gpt-2-output-dataset/james/MAUVE_datasets/mauve-human-eval-anon-human_clean.csv',
#     index=False)

