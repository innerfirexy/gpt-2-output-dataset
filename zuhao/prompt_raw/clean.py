import re
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize

'''
This script is to clean all the raw text from wikitext/truenews/story domains.
'''

##### wikitext_35.txt #####

# Set the maximum length
prompt_len = 35
text_len = 5000

# Load the text from the file
with open("/Users/yangzuhao/Downloads/wikitext-103/wikitext.txt", "r") as f:
    text = f.read()

# Define the regular expression pattern for section titles
pattern = r'\s*=\s*=\s*=\s*(.*?)\s*=\s*=\s*='

# Split the text into sections based on the pattern
sections = re.split(pattern, text)

# Initialize a list to store the 35-token sequences
result = []

# Iterate over sections and extract the 35-token sequences
print("Extracting articles......")
for section in tqdm(sections, total=len(sections)):
    # Tokenize the section using nltk's word_tokenize
    tokens = word_tokenize(section)

    # Check if there are at least 35 tokens
    if len(tokens) >= prompt_len:
        # Extract the first 35 tokens and join them back into a string
        short_content = " ".join(tokens[:35])
        if not short_content.startswith("="):
            result.append(short_content)

    # Break the loop if there are 5,000 articles
    if len(result) >= text_len:
        break

# Write the 35-token sequences to a new text file
print("Writing output......")
with open("/Users/yangzuhao/Downloads/wikitext_35.txt", "w") as f:
    for i, sequence in tqdm(enumerate(result, start=1)):
        f.write(f"Prompt {i}:\n")
        f.write(sequence)
        f.write("\n\n")


##### truenews_35.txt #####
        
# Load the CSV data into a DataFrame
df = pd.read_csv("/Users/yangzuhao/Downloads/wikinews.csv")

# Define a regular expression pattern for the source indication
source_pattern = r'^.*\(Reuters\) - '

# Initialize counters and the result list
counter_politics = 0
counter_world = 0
result = []

# Iterate over the rows in the DataFrame
print("Extracting articles......")
for _, row in tqdm(df.iterrows()):
    text = row["text"]
    subject = row["subject"]

    # Remove the source indication from the beginning of the text
    text = re.sub(source_pattern, "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Check if there are at least 35 tokens
    if len(tokens) >= 35:
        # Extract the first 35 tokens and join them back into a string
        short_content = " ".join(tokens[:35])

        # Append the short_content to the result list if it doesn't start with "="
        if not short_content.startswith("="):
            if subject == "politicsNews" and counter_politics < 2500:
                result.append(short_content)
                counter_politics += 1
            elif subject == "worldnews" and counter_world < 2500:
                result.append(short_content)
                counter_world += 1

    # Break the loop if there are 5,000 prompts
    if len(result) >= 5000:
        break

# Write the 35-token sequences to a new text file
print("Writing output......")
with open("/Users/yangzuhao/Downloads/truenews_35.txt", "w") as f:
    for i, sequence in tqdm(enumerate(result, start=1)):
        f.write(f"Prompt {i}:\n")
        f.write(sequence)
        f.write("\n\n")


##### story_vary.txt #####

# Define a regular expression pattern for the square bracket patterns
pattern = r'^(\[ [A-Za-z0-9]+ \] )+'

# Initialize the result list
result = []

# Open the input file and read the lines
with open("/Users/yangzuhao/Downloads/story.txt", "r") as f:
    lines = f.readlines()

# Iterate over the lines in the input file
print("Extracting articles......")
for line in tqdm(lines):
    # Remove the square bracket patterns from the beginning of the line
    cleaned_line = re.sub(pattern, "", line)

    # Append the cleaned_line to the result list
    result.append(cleaned_line)

    # Break the loop if there are 5,000 prompts
    if len(result) >= 5000:
        break

# Write the cleaned prompts to a new text file
print("Writing output......")
with open("/Users/yangzuhao/Downloads/story_vary.txt", "w") as f:
    for i, prompt in tqdm(enumerate(result, start=1)):
        f.write(f"Prompt {i}:\n")
        f.write(prompt)
        f.write("\n")
