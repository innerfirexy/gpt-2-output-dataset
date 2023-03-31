import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

path = '/Users/james/Workspace/gpt-2-output-dataset/data/'
tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_len_list = []
with open(path + 'webtext.test.jsonl') as f:
    df = pd.read_json(f, lines=True)
    for index, row in tqdm(df.iterrows()):
        token_len_list.append(len(
            tokenizer(row['text'], truncation=True,
                      max_length=1024)['input_ids']))

df['token_len'] = token_len_list
print(df.head())

df_sorted = df.sort_values(by='token_len', ascending=True)
df_sorted.to_json(path + 'webtext.test.model.sorted.jsonl',
                  orient='records',
                  lines=True)
