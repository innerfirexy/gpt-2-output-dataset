import pandas as pd
from transformers import AutoTokenizer

'''

This script is used to calculate the length of the text in the dataset only.

'''
for model_name in ["gpt2", "gpt2-xl"]:
    for split in ['story_vary', 'truenews_35', 'wikitext_35']:
        print(f"computing length for {model_name} {split}:")
        path = "../data/gpt2-generated-from-prompt/"
        with open(f"{path}{model_name}-{split}.jsonl") as f:
            df = pd.read_json(f, lines=True)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        result = [0] * 5
        for index, row in df.iterrows():
            text = row["gen_text"]
            text = tokenizer(text, truncation=True, max_length=1024)["input_ids"]
            # print(text, type(text), len(text))
            if len(text) >= 0 and len(text) < 200:
                result[0] += 1
            elif len(text) >= 200 and len(text) < 400:
                result[1] += 1
            elif len(text) >= 400 and len(text) < 600:
                result[2] += 1
            elif len(text) >= 600 and len(text) < 800:
                result[3] += 1
            elif len(text) >= 800 and len(text) <= 1024:
                result[4] += 1

        print(result, sum(result), model_name+" "+split)

# [549, 625, 296, 241, 3289] 5000 gpt2 story_vary
# [750, 1222, 824, 584, 1620] 5000 gpt2 truenews_35
# [403, 571, 251, 260, 3515] 5000 gpt2 wikitext_35
# [745, 757, 404, 324, 2770] 5000 gpt2-xl story_vary
# [836, 1336, 759, 678, 1391] 5000 gpt2-xl truenews_35
# [485, 672, 316, 310, 3217] 5000 gpt2-xl wikitext_35
