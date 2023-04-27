import pandas as pd
import json

# Open the file and read the JSON string
with open('result.json', 'r') as f:
    json_str = f.read()

# Convert the JSON string to a Python dictionary
data_dict = json.loads(json_str)

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)
print(df)
df['domain'] = ''
df['model'] = ''
df['split_len'] = ''

base_path = '../data/gpt2-generated-from-prompt'
categories = ['gs_story', 'gs_news', 'gs_wiki']
categories_wo_gs = ['story', 'news', 'wiki']
split_domain_list = ['story_vary', 'truenews_35', 'wikitext_35']
types = ['human', 'gen', 'pair']

file_template = "../{cat}/{cat_wo_gs}_{idx}.jsonl"
split_template = "{base}/split_new/{model}-{split}.sorted.split.{idx}.jsonl"
pair_template = "{base}/split_new/{model}-{split}.sorted.split.{idx}.pair.jsonl"

file_lists = {t: [] for t in types}
length_list = [200, 400, 600, 800, 1024] * 6

count = 0
for index, cat in enumerate(categories):
    cat_wo_gs = categories_wo_gs[index]
    for i in range(5):
        for model_name in ['gpt2', 'gpt2-xl']:
            df.iloc[count, df.columns.get_loc('domain')] = cat_wo_gs
            df.iloc[count, df.columns.get_loc('model')] = model_name
            df.iloc[count, df.columns.get_loc('split_len')] = i*200
            count += 1
            # file_lists['human'].append(file_template.format(cat=cat, cat_wo_gs=cat_wo_gs, idx=i))
            # file_lists['gen'].append(split_template.format(base=base_path, model=model_name, split=split_domain, idx=i*200))
            # file_lists['pair'].append(pair_template.format(base=base_path, model=model_name, split=split_domain, idx=i*200))

print(df)
df.to_csv('result_labelled.csv', index=False)