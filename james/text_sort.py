import pandas as pd

path  = '/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/5273_sample/'
with open(path + 'webtext.train.jsonl') as f:
    df = pd.read_json(f, lines=True)
    
print(df.head())

df_sorted = df.sort_values(by='text', key= lambda x: x.str.len())
df_sorted.to_json(path + 'webtext.train.sorted.jsonl', orient='records', lines=True)