import pandas as pd
from transformers import AutoTokenizer

'''
This script is used to calculate the length of the text in the dataset only.
'''

path = "/root/autodl-tmp/gpt-2-output-dataset/data/"
# postfix = 'valid' # train_old /valid /test
postfix = 'train_opt_125m_top_50_wiki' # train_opt_size_sampling_method_domain
with open(path + f"webtext.{postfix}.jsonl") as f:
    df = pd.read_json(f, lines=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

result = [0] * 5
for index, row in df.iterrows():
    text = row["text"]
    text = tokenizer(text, truncation=True, max_length=1024)["input_ids"]
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

print(result, sum(result))

# [996, 907, 736, 523, 1838]  5000 Webtext_valid
# [1002, 944, 751, 526, 1777] 5000 Webtext_test
# [2731, 715, 241, 160, 1153] 5000 OPT_125m_Story
# [964, 888, 441, 268, 2439]  5000 OPT_125m_Wiki
# [844, 1220, 1194, 764, 978] 5000 OPT_125m_News

"""
# Make up for improper strings
import json

def fix_invalid_quotes(json_str):
    temporarily_fixed_json = json_str.replace('\\"', 'TEMP_ESCAPE_SEQUENCE')
    fixed_json = temporarily_fixed_json.replace('"', '\\"')
    fixed_json = fixed_json.replace('\\"{', '{')
    fixed_json = fixed_json.replace('}\\"', '}')
    fixed_json = fixed_json.replace('TEMP_ESCAPE_SEQUENCE', '\\"')
    return fixed_json

input_file = path + f"webtext.{postfix}.jsonl"
output_file = path + "webtext.train_new.jsonl"

# Open the input file for reading and the output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Iterate through each line in the input file
    for line in infile:
        # Remove the newline character at the end of the line
        line = line.strip()
        # Fix invalid quotes in the JSON string
        fixed_line = fix_invalid_quotes(line)
        try:
            # Load the JSON object from the fixed line
            json_obj = json.loads(fixed_line)
            # Perform any replacements or modifications to the JSON object if needed
            # Write the modified JSON object as a line in the output file
            outfile.write(json.dumps(json_obj) + '\n')
        except json.JSONDecodeError as e:
            # print(f"Failed to decode JSON on line: {fixed_line}")
            # print(f"Error: {e}")
            continue
"""