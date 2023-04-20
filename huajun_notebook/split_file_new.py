import pandas as pd
from transformers import AutoTokenizer, GPT2Tokenizer
import jsonlines
from tqdm import tqdm

path = '../data/gpt2-generated-from-prompt/'
gen_path = '../data/gpt2-generated-from-prompt/split_new/'

for model_name in ["gpt2", "gpt2-xl"]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for split in ['story_vary', 'truenews_35', 'wikitext_35']:
        filename = model_name + '-' + split + '.sorted.jsonl'
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