import json
import numpy as np
import torch
from tqdm import tqdm
import mauve
from simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import operator
from scipy import stats
from Bleu import Bleu
from SelfBleu import SelfBleu

import os
# Set TOKENIZERS_PARALLELISM to 'true' or 'false' to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


###############
# Data Loader #
###############
def load_gpt2_dataset_old(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append(json.loads(line)["text"])
        except json.decoder.JSONDecodeError: # skip to next line when encountering wrong string format
            continue
   
    return texts # [gen_text1, gen_text2, ...]

def load_gpt2_dataset(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    with open(json_file_name, encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            try:
                obj = json.loads(line)
                if "text" in obj:
                    texts.append(obj["text"])
                elif "gen_text" in obj:
                    texts.append(obj["gen_text"])
                else:
                    print(f"Neither 'text' nor 'gen_text' found in line {i} of {json_file_name}")
                # texts.append(json.loads(line)["text"])
            except json.decoder.JSONDecodeError: # skip to next line when encountering wrong string format
                continue
   
    return texts # [gen_text1, gen_text2, ...]


def load_gpt2_pair(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append((json.loads(line)["prompt_text"], json.loads(line)["gen_text"]))
        except json.decoder.JSONDecodeError: # skip to next line when encountering wrong string format
            continue

    return texts # [(prompt_text1, gen_text1), (prompt_text2, gen_text2), ...]


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
    
    Results: webtext_train(9310) vs. opt_13b-125m(5273) --> mauve = 0.8627802521742031 (1024) / 0.9102370353331146 (256) / 0.9266521248714534 (128)
             webtext_valid(5000) vs. opt_13b-125m(5273) --> mauve = 0.8735936367558702 (1024) / 0.9218938670439518 (256) / 0.9272530776764185 (128)
             webtext_test(5000)  vs. opt_13b-125m(5273) --> mauve = 0.8552726757055065 (1024) / 0.9187265446563135 (256) / 0.9203297293141568 (128)
    Finding: The lower tgt_len, the better performance.
    """
    
    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to {tgt_len} tokens
    mauve_score = mauve.compute_mauve(p_text=human_text, q_text=gen_text, device_id=0, max_text_length=max_len, verbose=False, featurize_model_name="gpt2").mauve
    
    return mauve_score


##########################
# Repetition & Diversity #
##########################
def compute_rep_div(gen_text):
    """
    Compute the rep-2, rep-3, rep-4 and diversity scores of given text.
    
    :param gen_text: model-generated text
    :return rep_2: 2-gram repetition score of given text
    :return rep_3: 3-gram repetition score of given text
    :return rep_4: 4-gram repetition score of given text
    :return div_score: diversity score of given text
    
    Results: rep-2 score: 8.25 [with batch_decode]
             rep-3 score: 4.17
             rep-4 score: 2.95
             diversity score: 0.8533026626250001
    Finding: The lower tgt_len, the better performance. batch_decode is involved as it does help produce better scores.
    """
    
    rep_2, rep_3, rep_4, div_score = measure_repetition_and_diversity(gen_text)
    
    return rep_2, rep_3, rep_4, div_score


#############
# Coherence #
#############
def compute_coh(file_name):
    """
    Compute the coherence score of given text with reference to its prefix.
    
    :param file_name: jsonl file which stores <prompt_text, gen_text> pairs
    :return coh_score: coherence score of given text with reference to its prefix
    
    Results: coherence score: 0.8022059978309312 [w/o batch_decode]
    Finding: Indepedent of {tgt_len}.
    """
    
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sent_lst = load_gpt2_pair(file_name) # "opttext_pair.jsonl"
    pp_lst, yy_lst = zip(*sent_lst)
    similarities = np.array(model.similarity(list(pp_lst), list(yy_lst)))
    coh_score = similarities.trace() / len(similarities) 
    
    return coh_score


########
# BLEU #
########
def compute_bleu(human_text, gen_text):
    """
    Compute the BLEU score of given text with reference to webtext.
    
    :param human_text: human text (webtext)
    :param gen_text: model-generated text
    :return bleu_score: BLEU score of given text with reference to webtext
    
    Results: webtext_train(9310) vs. opt_13b-125m(5273) --> bleu = 0.3199899896719573 (1024) / 0.2445160862568812 (256) / 0.20247434364644207 (128) [with batch_decode]
             webtext_valid(5000) vs. opt_13b-125m(5273) --> bleu = 0.3174006504263524 (1024) / 0.24456896382229462 (256) / 0.2013412137537578 (128)
             webtext_test(5000)  vs. opt_13b-125m(5273) --> bleu = 0.3174536690014808 (1024) / 0.24404910401572574 (256) / 0.20054597183508927 (128)
    Finding: The higher tgt_len, the better performance.
    """
    
    bleu = Bleu()
    bleu.real_data = human_text # human text
    bleu.test_data = gen_text # model text
    bleu_score = bleu.get_score()
    
    return bleu_score


#############
# Self-BLEU #
#############
def compute_self_bleu(gen_text):
    """
    Compute the Self-BLEU score of given text.
    
    :param gen_text: model-generated text
    :return self_bleu_score: Self-BLEU score of given text
    
    Results: 0.3786205616072991 (1024) / 0.372401794537256 (256) / 0.3283484372757455 (128) [with batch_decode]
    Finding: The higher tgt_len, the better performance.
    """
    
    self_bleu = SelfBleu()
    self_bleu.test_data = gen_text # model text
    self_bleu_score = self_bleu.get_score()
    
    return self_bleu_score


##############
# Perplexity #
##############
def compute_perplexity(gen_text, max_len, batch_size):
    """
    Compute the perplexity score of given text.
    
    :param gen_text: model-generated text
    :param max_len: maximum text length to truncate
    :param batch_size: batch size used in perplexity computation
    :return perplexity_score: perplexity score of given text
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2").half().cuda()
    text_list = []
    print("Processing text......")
    for text in tqdm(gen_text, total=len(gen_text)):
        temp_sent = text.replace(' @', '').replace('@ ', '') # remove space 
        temp_sent = TreebankWordDetokenizer().detokenize(temp_sent.split())
        text_list.append(temp_sent)
    score_list = []
    print("Computing perplexity score......")
    for i in tqdm(range(len(text_list) // batch_size), total=len(text_list) // batch_size):
        text_list_i = text_list[i * batch_size:(i+1) * batch_size]
        inputs = tokenizer(text_list_i, return_tensors='pt', truncation=True, padding=True, max_length=max_len)
        with torch.no_grad():
            labels = inputs['input_ids'].cuda() 
            labels[labels == tokenizer.pad_token] = -100 
            out = lm(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), 
                     labels=labels)
            score_list.append(out.loss) 
    score_list = torch.tensor(score_list)
    ppl = np.e ** score_list.mean()
    perplexity_score = ppl.item()
    
    return perplexity_score


####################
# Zipf Coefficient #
####################
def compute_zipf(gen_text, max_len):
    """
    Compute the zipf score of given text.
    
    :param gen_text: model-generated text
    :param max_len: maximum text length to truncate
    :return zipf_score: zipf score of given text
    """
    cnt = Counter()
    
    print("Counting token frequency......")
    for line in tqdm(gen_text, total=len(gen_text)):
        gen = word_tokenize(line)
        cnt.update(gen)

    xs = np.arange(1, min(len(cnt), max_len)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:max_len])
    zipf_score, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    print("zipf value\tregression r value\tregression p value")
    print(f"{-zipf_score}\t{-r}\t{p}")
    
    return -zipf_score

def retrieve_model_name(file_path):
    allowed_models = ['gpt2', 'gpt2-xl']
    
    file_name = file_path.split("/")[-1]  # Get the file name without the directory path
    model_name = file_name.split("-")[0]  # Get the model name before the hyphen
    
    if model_name not in allowed_models:
        raise ValueError(f"Invalid model name '{model_name}'. Allowed models are: {', '.join(allowed_models)}")
    
    return model_name


if __name__ == "__main__":

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
    
    for index, cat in enumerate(categories):
        split_domain = split_domain_list[index]
        cat_wo_gs = categories_wo_gs[index]
        for i in range(5):
            file_lists['human'].append(file_template.format(cat=cat, cat_wo_gs=cat_wo_gs, idx=i))
            
            for model_name in ['gpt2', 'gpt2-xl']:
                file_lists['gen'].append(split_template.format(base=base_path, model=model_name, split=split_domain, idx=i*200))
                file_lists['pair'].append(pair_template.format(base=base_path, model=model_name, split=split_domain, idx=i*200))

    result = [{} for _ in range(30)]

    for i in tqdm(range(30), total=30):
        # basic info
        print(f"Human Reference: {file_lists['human'][i]}")
        print(f"Model-generated Text: {file_lists['gen'][i]}")
        print(f"Length Interval: {length_list[i]}")
        print(' ------------------------ Divide Line ------------------------')

        # hyper-parameters
        tgt_len = length_list[i]
        batch_size = 20

        # load original human & model texts
        p_text_ = load_gpt2_dataset(file_lists['human'][i])
        q_text_ = load_gpt2_dataset(file_lists['gen'][i])

        # tokenization & batch_decode
        model_name = retrieve_model_name(file_lists['gen'][i])
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        x = tokenizer(p_text_, truncation=True, max_length=tgt_len)["input_ids"]
        y = tokenizer(q_text_, truncation=True, max_length=tgt_len)["input_ids"]
        print("Performing batch_decode......")
        xxyy = [(xx, yy) for (xx, yy) in tqdm(zip(x, y), total=min(len(x), len(y))) if len(xx) <= tgt_len and len(yy) <= tgt_len]
        x, y = zip(*xxyy)

        # map back to texts
        p_text = tokenizer.batch_decode(x) # [:target_num]
        q_text = tokenizer.batch_decode(y) # [:target_num]

        # compute scores
        mauve_score = compute_mauve(p_text, q_text, tgt_len)
        print("mauve score:", mauve_score)
        result[i]["mauve"] = mauve_score

        rep_2, rep_3, rep_4, div_score = compute_rep_div(q_text_)
        print("rep-2 score:", rep_2)
        result[i]["rep-2"] = rep_2
        print("rep-3 score:", rep_3)
        result[i]["rep-3"] = rep_3
        print("rep-4 score:", rep_4)
        result[i]["rep-4"] = rep_4
        print("diversity score:", div_score)
        result[i]["diversity"] = div_score

        coh_score = compute_coh(file_name=file_lists['pair'][i])
        print("coherence score:", coh_score)
        result[i]["coherence"] = coh_score

        bleu_score = compute_bleu(p_text_, q_text_)
        print("bleu score:", bleu_score)
        result[i]["bleu"] = bleu_score

        self_bleu_score = compute_self_bleu(q_text_)
        print("self-bleu score:", self_bleu_score)
        result[i]["self-bleu"] = self_bleu_score

        perplexity_score = compute_perplexity(q_text_, tgt_len, batch_size)
        print("perplexity score:", perplexity_score)
        result[i]["perplexity"] = perplexity_score

        zipf_score = compute_zipf(q_text_, tgt_len)
        print("zipf score:", zipf_score)
        result[i]["zipf"] = zipf_score

        print(f"results for the {i+1}-th iteration: {result[i]}")

    # output results
    with open("result.txt", "w") as f:
        f.write(json.dumps(result))
