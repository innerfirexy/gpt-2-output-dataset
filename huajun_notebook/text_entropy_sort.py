import pandas as pd
from transformers import AutoTokenizer, GPT2Tokenizer
from tqdm import tqdm

path = '../data/gpt2-generated-from-prompt/'
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

for model_name in ["gpt2", "gpt2-xl"]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for split in ['story_vary', 'truenews_35', 'wikitext_35']:
        print(f"computing for {model_name} {split}:")
        entropy_list = []
        with open(path + f'model={model_name}-{split}.nll') as fe:
            for line in fe:
                entropy_list.append(list(map(float, line.strip().split())))

        token_len_list = []
        with open(f'{path}{model_name}-{split}.jsonl') as f:
            df = pd.read_json(f, lines=True)
            for index, row in tqdm(df.iterrows()):
                token_len_list.append(
                    len(
                        tokenizer(row['gen_text'], truncation=True,
                                max_length=1024)['input_ids']))

        df['token_len'] = token_len_list
        df['entropy'] = entropy_list
        print(df.head())

        df_sorted = df.sort_values(by='token_len', ascending=True)
        df_sorted.to_json(f'{path}{model_name}-{split}.sorted.jsonl',
                        orient='records',
                        lines=True)
