import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

path = '/root/autodl-tmp/gpt-2-output-dataset/data/'
# postfix = 'valid' # train_old /valid /test
postfix = 'train_opt_125m_top_50_wiki' # train_opt_size_sampling_method_domain
tokenizer = AutoTokenizer.from_pretrained("gpt2")

entropy_list = []
with open(path + f'webtext.{postfix}.model=.nll') as fe:
    for line in fe:
        entropy_list.append(list(map(float, line.strip().split())))

token_len_list = []
with open(path + f'webtext.{postfix}.jsonl') as f:
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
df_sorted.to_json(path + f'webtext.{postfix}.sorted.jsonl',
                  orient='records',
                  lines=True)
