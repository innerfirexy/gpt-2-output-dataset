import pandas as pd
from transformers import AutoTokenizer, GPT2Tokenizer
import jsonlines
from tqdm import tqdm

path = '../data/gpt2-generated-from-prompt/'
gen_path = '../data/gpt2-generated-from-prompt/split/'
# gen_path = "/root/autodl-tmp/gpt-2-output-dataset/data/"
for model_name in ["gpt2", "gpt2-xl"]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for split in ['story_vary', 'truenews_35', 'wikitext_35']:
        postfix = model_name + '-' + split
        # postfix = 'train_opt_125m_top_50_wiki' # train_opt_size_sampling_method_domain
        with open(f'{path}{postfix}.sorted.jsonl') as f:
            df = pd.read_json(f, lines=True)

        var_len = 0
        gap = 200
        for index, row in tqdm(df.iterrows()):
            if row['token_len'] >= var_len and row["token_len"] < var_len + gap:
                with jsonlines.open(
                        gen_path + f"{postfix}.split_" + str(var_len) + ".jsonl",
                        "a") as w:
                    w.write({'text': row["gen_text"]})
                with open(
                        gen_path + f"{postfix}.split_" + str(var_len) + ".nll",
                        "a") as we:
                    entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
                    we.write(f'{entropy}\n')
            else:
                var_len += 200
                if var_len + gap == 1000:
                    gap = 225
                with jsonlines.open(
                        gen_path + f"{postfix}.split_" + str(var_len) + ".jsonl",
                        "a") as w:
                    w.write({'text': row["gen_text"]})
                with open(
                        gen_path + f"{postfix}.split_" + str(var_len) + ".nll",
                        "a") as we:
                    entropy = ' '.join(f'{num:.4f}' for num in row["entropy"])
                    we.write(f'{entropy}\n')
            