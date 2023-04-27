import json
from tqdm import tqdm


# Get the prompt_text
prompt_dict = {}
with open("../zuhao/prompt_raw/story_vary.txt", "r", encoding='utf-8') as file:
    content_story = file.readlines()
prompt_dict["story_vary"] = [content_story[i].rstrip('\n') for i in range(1, 15000, 3)]

with open("../zuhao/prompt_raw/truenews_35.txt", "r", encoding='utf-8') as file:
    content_news = file.readlines()
prompt_dict["truenews_35"] = [content_news[i].rstrip('\n') for i in range(1, 15000, 3)]

with open("../zuhao/prompt_raw/wikitext_35.txt", "r", encoding='utf-8') as file:
    content_wiki = file.readlines()
prompt_dict["wikitext_35"] = [content_wiki[i].rstrip('\n') for i in range(1, 15000, 3)]

# Set the file path
gen_path = '../data/gpt2-generated-from-prompt/split_new/'
for model in ["gpt2", "gpt2-xl"]:
    for split in ['story_vary', 'truenews_35', 'wikitext_35']:
        for split_len in range(0, 1000, 200):
            # Separate prompt_text and gen_text
            filename = f"{model}-{split}.sorted.split.{split_len}.jsonl"
            print(f"current loop: {filename}")
            out_list = []
            with open(gen_path + filename, "r", encoding='utf-8') as file:
                print("Separating prompt_text and gen_text......")
                for line in tqdm(file):
                    idx = json.loads(line)["id"]
                    ended = json.loads(line)["ended"]
                    length = json.loads(line)["length"]
                    text = json.loads(line)["gen_text"]
                    prompt_text = prompt_dict[split][idx]
                    text1 = text[:len(prompt_text)-1]
                    text2 = text[len(prompt_text):]  
                    out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
                    out_list.append(out)
                    
            with open(gen_path + f"{model}-{split}.sorted.split.{split_len}.pair.jsonl", "w") as file:
                print("Outputing paired jsonl file......")
                for out in tqdm(out_list):
                    json.dump(out, file)
                    file.write("\n")
        
# out_news = []
# with open(f"/root/autodl-tmp/gpt-2-output-dataset/data/webtext.train_opt_{model}_top_50_news.sorted.split.{split}.jsonl", "r") as file:
#     print("Separating prompt_text and gen_text......")
#     for line in tqdm(file):
#         idx = json.loads(line)["id"]
#         ended = json.loads(line)["ended"]
#         length = json.loads(line)["length"]
#         text = json.loads(line)["text"]
#         prompt_text = prompt_dict["news"][idx]
#         text1 = text[:len(prompt_text)-1]
#         text2 = text[len(prompt_text):]  
#         out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
#         out_news.append(out)
        
# with open(f"/root/autodl-tmp/gpt-2-output-dataset/pair/webtext.train_opt_{model}_top_50_news.sorted.split.{split}.pair.jsonl", "w") as file:
#     print("Outputing paired jsonl file......")
#     for out in tqdm(out_news):
#         json.dump(out, file)
#         file.write("\n")

# out_wiki = []
# with open(f"/root/autodl-tmp/gpt-2-output-dataset/data/webtext.train_opt_{model}_top_50_wiki.sorted.split.{split}.jsonl", "r") as file:
#     print("Separating prompt_text and gen_text......")
#     for line in tqdm(file):
#         idx = json.loads(line)["id"]
#         ended = json.loads(line)["ended"]
#         length = json.loads(line)["length"]
#         text = json.loads(line)["text"]
#         prompt_text = prompt_dict["wiki"][idx]
#         text1 = text[:len(prompt_text)-1]
#         text2 = text[len(prompt_text):]  
#         out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
#         out_wiki.append(out)
        
# with open(f"/root/autodl-tmp/gpt-2-output-dataset/pair/webtext.train_opt_{model}_top_50_wiki.sorted.split.{split}.pair.jsonl", "w") as file:
#     print("Outputing paired jsonl file......")
#     for out in tqdm(out_wiki):
#         json.dump(out, file)
#         file.write("\n")
