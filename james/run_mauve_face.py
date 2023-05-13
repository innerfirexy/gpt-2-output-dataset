import json
import numpy as np
from tqdm import tqdm
import mauve
from transformers import AutoTokenizer
from simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE
from Bleu import Bleu
from SelfBleu import SelfBleu

import os

# Set TOKENIZERS_PARALLELISM to 'true' or 'false' to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


###############
# Data Loader #
###############
def load_gpt2_dataset(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append(json.loads(line)["text"])
        except json.decoder.JSONDecodeError:  # skip to next line when encountering wrong string format
            continue

    return texts  # [gen_text1, gen_text2, ...]


def load_get2_pair(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append((json.loads(line)["prompt"],
                          json.loads(line)["text"]))
        except json.decoder.JSONDecodeError:  # skip to next line when encountering wrong string format
            continue

    return texts  # [(prompt_text1, gen_text1), (prompt_text2, gen_text2), ...]


#########
# MAUVE #
#########
def compute_mauve(human_text, gen_text, max_len):
    """
    Compute the MAUVE score of given text with reference to webtext.
    
    :param human_text: human text (webtext)
    :param gen_text: model-generated text
    :param max_len: maximum text length to truncate
    :return mauve_score: MAUVE score of given text with reference to webtext
    
    """

    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to {tgt_len} tokens
    mauve_score = mauve.compute_mauve(
        p_text=human_text,
        q_text=gen_text,
        device_id=0,
        max_text_length=max_len,
        verbose=False,
        featurize_model_name="gpt2",
    ).mauve

    return mauve_score



if __name__ == "__main__":
    # hyper-parameters
    tgt_len = 256   # max text length (1024 / 256 / 128); 128 is used in Contrastive Decoding code
    split = "train"  # reference data source (train / valid / test)

    # load original human & model texts
    p_text_ = load_gpt2_dataset(
        "/home/yyuan/gpt-2-output-dataset/james/wiki/wiki_4.jsonl"
    )  # human text
    q_text_ = load_gpt2_dataset(
        "/home/yyuan/gpt-2-output-dataset/james/split_wiki/webtext.train.model=.bloom_7b1.wiki.sorted.split.800.jsonl"
    )  # model text

    # tokenization & batch_decode
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    x = tokenizer(p_text_, truncation=True, max_length=tgt_len)["input_ids"]
    y = tokenizer(q_text_, truncation=True, max_length=tgt_len)["input_ids"]
    print("Performing batch_decode......")
    xxyy = [(xx, yy) for (xx, yy) in tqdm(zip(x, y), total=min(len(x), len(y)))
            if len(xx) <= tgt_len and len(yy) <= tgt_len]
    x, y = zip(*xxyy)

    # map back to texts
    p_text = tokenizer.batch_decode(x)  # [:target_num]
    q_text = tokenizer.batch_decode(y)  # [:target_num]

    # compute scores
    mauve_score = compute_mauve(p_text, q_text, tgt_len)
    print("mauve score:", mauve_score)

