import re
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
import jsonlines
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
'''
This script is to clean all the raw tokens from wikitokens/truenews/story domains.
'''


def extract_golden_story():
    targets = [0] * 5
    file = '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story.target.txt'
    with open(file) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = re.sub(r"\*", "", line)
        line = re.sub("<newline>", "\n", line)
        # print(type(line))
        tokens = tokenizer(line, max_length=10000)["input_ids"]
        if len(tokens) >= 0 and len(tokens) < 200:
            targets[0] += 1
            if targets[0] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story_0.jsonl',
                        'a') as f0:
                    f0.write({'text': line, 'token_len': len(tokens)})
        elif len(tokens) >= 200 and len(tokens) < 400:
            targets[1] += 1
            if targets[1] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story_1.jsonl',
                        'a') as f1:
                    f1.write({'text': line, 'token_len': len(tokens)})
        elif len(tokens) >= 400 and len(tokens) < 600:
            targets[2] += 1
            if targets[2] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story_2.jsonl',
                        'a') as f2:
                    f2.write({'text': line, 'token_len': len(tokens)})
        elif len(tokens) >= 600 and len(tokens) < 800:
            targets[3] += 1
            if targets[3] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story_3.jsonl',
                        'a') as f3:
                    f3.write({'text': line, 'token_len': len(tokens)})
        elif len(tokens) >= 800 and len(tokens) <= 1024:
            targets[4] += 1
            if targets[4] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/story_4.jsonl',
                        'a') as f4:
                    f4.write({'text': line, 'token_len': len(tokens)})
        else:
            pass


def compute_len_dist_news():
    len_dist = [0] * 5
    file = '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wikinews.csv'
    df = pd.read_csv(file)
    df = df[df['subject'] == 'worldnews']
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        source_pattern = r'^.*\(Reuters\) - '
        tokens = tokenizer(row['text'], truncation=True,
                           max_length=10000)["input_ids"]
        if len(tokens) >= 0 and len(tokens) < 200:
            len_dist[0] += 1
        elif len(tokens) >= 200 and len(tokens) < 400:
            len_dist[1] += 1
        elif len(tokens) >= 400 and len(tokens) < 600:
            len_dist[2] += 1
        elif len(tokens) >= 600 and len(tokens) < 800:
            len_dist[3] += 1
        elif len(tokens) >= 800 and len(tokens) <= 1024:
            len_dist[4] += 1
        else:
            pass
    print(len_dist, sum(len_dist))


def extract_golden_news():
    file = '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wikinews.csv'
    df = pd.read_csv(file)
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        source_pattern = r'^.*\(Reuters\) - '
        text = re.sub(source_pattern, "", row['text'])
        tokens = tokenizer(text, truncation=True,
                           max_length=10000)["input_ids"]
        if len(tokens) >= 0 and len(tokens) < 200:
            with jsonlines.open(
                    '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/news_0.jsonl',
                    'a') as f0:
                f0.write({'text': text, 'token_len': len(tokens)})
        elif len(tokens) >= 200 and len(tokens) < 400:
            with jsonlines.open(
                    '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/news_1.jsonl',
                    'a') as f1:
                f1.write({'text': text, 'token_len': len(tokens)})
        elif len(tokens) >= 400 and len(tokens) < 600:
            with jsonlines.open(
                    '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/news_2.jsonl',
                    'a') as f2:
                f2.write({'text': text, 'token_len': len(tokens)})
        elif len(tokens) >= 600 and len(tokens) < 800:
            with jsonlines.open(
                    '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/news_3.jsonl',
                    'a') as f3:
                f3.write({'text': text, 'token_len': len(tokens)})
        elif len(tokens) >= 800 and len(tokens) <= 1024:
            with jsonlines.open(
                    '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/news_4.jsonl',
                    'a') as f4:
                f4.write({'text': text, 'token_len': len(tokens)})
        else:
            pass


def extract_golden_wiki():
    targets = [0] * 5
    extract_texts_list = []
    path = '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wikitext.txt'
    with open(path) as f:
        f = f.read()
    texts = f.split('\n')
    section = ''
    for text in tqdm(texts, total=len(texts)):
        if text == '\n' or text == ' ':
            continue
        if text.startswith(' ='):
            extract_texts_list.append(section)
            section = ''
            continue
        else:
            section += text

    for i in tqdm(extract_texts_list):
        i = i.strip()
        tokens = tokenizer(i, max_length=10000)["input_ids"]
        if len(tokens) >= 35 and len(tokens) < 200:
            targets[0] += 1
            if targets[0] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wiki_0.jsonl',
                        'a') as f0:
                    f0.write({'text': i, 'token_len': len(tokens)})
        elif len(tokens) >= 200 and len(tokens) < 400:
            targets[1] += 1
            if targets[1] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wiki_1.jsonl',
                        'a') as f1:
                    f1.write({'text': i, 'token_len': len(tokens)})
        elif len(tokens) >= 400 and len(tokens) < 600:
            targets[2] += 1
            if targets[2] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wiki_2.jsonl',
                        'a') as f2:
                    f2.write({'text': i, 'token_len': len(tokens)})
        elif len(tokens) >= 600 and len(tokens) < 800:
            targets[3] += 1
            if targets[3] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wiki_3.jsonl',
                        'a') as f3:
                    f3.write({'text': i, 'token_len': len(tokens)})
        elif len(tokens) >= 800 and len(tokens) <= 1024:
            targets[4] += 1
            if targets[4] <= 5000:
                with jsonlines.open(
                        '/Users/james/Workspace/gpt-2-output-dataset/golden_reference/wiki_4.jsonl',
                        'a') as f4:
                    f4.write({'text': i, 'token_len': len(tokens)})
        else:
            pass


if __name__ == "__main__":
    # extract_golden_story()
    # compute_len_dist_news()
    # extract_golden_news()
    extract_golden_wiki()
