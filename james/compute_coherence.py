import json
import numpy as np
import torch
from tqdm import tqdm
import mauve
from transformers import AutoTokenizer, AutoModelForCausalLM
from simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE
from Bleu import Bleu
from SelfBleu import SelfBleu
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from scipy import stats
import re
import os
# Set TOKENIZERS_PARALLELISM to 'true' or 'false' to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_get2_pair(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        authentic_gen_text = json.loads(line)["text"].replace(
            json.loads(line)["prompt"], '', 1)
        try:
            texts.append((json.loads(line)["prompt"], authentic_gen_text))
        except json.decoder.JSONDecodeError:  # skip to next line when encountering wrong string format
            continue

    return texts  # [(prompt_text1, gen_text1), (prompt_text2, gen_text2), ...]


def compute_coh(file_name):
    """
    Compute the coherence score of given text with reference to its prefix.
    
    :param file_name: jsonl file which stores <prompt_text, gen_text> pairs
    :return coh_score: coherence score of given text with reference to its prefix
    
    Results: coherence score: 0.8022059978309312 [w/o batch_decode]
    Finding: Indepedent of {tgt_len}.
    """

    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sent_lst = load_get2_pair(file_name)  # "opttext_pair.jsonl"
    pp_lst, yy_lst = zip(*sent_lst)
    similarities = np.array(model.similarity(list(pp_lst), list(yy_lst)))
    coh_score = similarities.trace() / len(similarities)

    return coh_score


if __name__ == "__main__":

    gen_text_list = [
        '/home/yyuan/gpt-2-output-dataset/james/split_story/webtext.train.model=.bloom_7b1.story.sorted.split.0.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_story/webtext.train.model=.bloom_7b1.story.sorted.split.200.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_story/webtext.train.model=.bloom_7b1.story.sorted.split.400.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_story/webtext.train.model=.bloom_7b1.story.sorted.split.600.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_story/webtext.train.model=.bloom_7b1.story.sorted.split.800.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.0.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.200.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.400.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.600.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.800.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.0.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.200.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.400.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.600.jsonl',
        '/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.800.jsonl',
    ]

    length_list = [200, 400, 600, 800, 1024] * 3

    for i in range(15):
        print(gen_text_list[i])
        print(length_list[i])
        print(' -------- Divide Line --------')

        tgt_len = length_list[i]
        batch_size = 20  # used in perplexity computation

        coh_score = compute_coh(file_name=gen_text_list[i])
        print("coherence score:", coh_score)