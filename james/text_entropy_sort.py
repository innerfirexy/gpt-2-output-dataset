import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

path = '/home/yyuan/gpt-2-output-dataset/data/'
tokenizer = AutoTokenizer.from_pretrained("gpt2")

entropy_list = []
with open(path + 'webtext.train.model=.bloom_7b1.wiki.nll') as fe:
    for line in fe:
        entropy_list.append(list(map(float, line.strip().split())))

token_len_list = []
with open(path + 'webtext.train.bloom_7b1.wiki.jsonl') as f:
    df = pd.read_json(f, lines=True)
    for index, row in tqdm(df.iterrows()):
        token_len_list.append(
            len(
                tokenizer(row['text'], truncation=True,
                          max_length=1024)['input_ids']))

df['token_len'] = token_len_list
df['entropy'] = entropy_list
print(df.head())

df_sorted = df.sort_values(by='token_len', ascending=True)
df_sorted.to_json(path + 'webtext.train.model=.bloom_7b1.wiki.sorted.jsonl',
                  orient='records',
                  lines=True)
