import json
from tqdm import tqdm


# Get the prompt_text
prompt_dict = {}
with open("story_vary.txt", "r") as file:
    content_story = file.readlines()
prompt_dict["story"] = [content_story[i].rstrip('\n') for i in range(1, 15000, 3)]

with open("truenews_35.txt", "r") as file:
    content_news = file.readlines()
prompt_dict["news"] = [content_news[i].rstrip('\n') for i in range(1, 15000, 3)]

with open("wikitext_35.txt", "r") as file:
    content_wiki = file.readlines()
prompt_dict["wiki"] = [content_wiki[i].rstrip('\n') for i in range(1, 15000, 3)]

# Set the file path
model = "125m"
split = 800

# Separate prompt_text and gen_text
out_story = []
with open(f"/root/autodl-tmp/gpt-2-output-dataset/data/webtext.train_opt_{model}_top_50_story.sorted.split.{split}.jsonl", "r") as file:
    print("Separating prompt_text and gen_text......")
    for line in tqdm(file):
        idx = json.loads(line)["id"]
        ended = json.loads(line)["ended"]
        length = json.loads(line)["length"]
        text = json.loads(line)["text"]
        prompt_text = prompt_dict["story"][idx]
        text1 = text[:len(prompt_text)-1]
        text2 = text[len(prompt_text):]  
        out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
        out_story.append(out)
        
with open(f"/root/autodl-tmp/gpt-2-output-dataset/pair/webtext.train_opt_{model}_top_50_story.sorted.split.{split}.pair.jsonl", "w") as file:
    print("Outputing paired jsonl file......")
    for out in tqdm(out_story):
        json.dump(out, file)
        file.write("\n")
        
out_news = []
with open(f"/root/autodl-tmp/gpt-2-output-dataset/data/webtext.train_opt_{model}_top_50_news.sorted.split.{split}.jsonl", "r") as file:
    print("Separating prompt_text and gen_text......")
    for line in tqdm(file):
        idx = json.loads(line)["id"]
        ended = json.loads(line)["ended"]
        length = json.loads(line)["length"]
        text = json.loads(line)["text"]
        prompt_text = prompt_dict["news"][idx]
        text1 = text[:len(prompt_text)-1]
        text2 = text[len(prompt_text):]  
        out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
        out_news.append(out)
        
with open(f"/root/autodl-tmp/gpt-2-output-dataset/pair/webtext.train_opt_{model}_top_50_news.sorted.split.{split}.pair.jsonl", "w") as file:
    print("Outputing paired jsonl file......")
    for out in tqdm(out_news):
        json.dump(out, file)
        file.write("\n")

out_wiki = []
with open(f"/root/autodl-tmp/gpt-2-output-dataset/data/webtext.train_opt_{model}_top_50_wiki.sorted.split.{split}.jsonl", "r") as file:
    print("Separating prompt_text and gen_text......")
    for line in tqdm(file):
        idx = json.loads(line)["id"]
        ended = json.loads(line)["ended"]
        length = json.loads(line)["length"]
        text = json.loads(line)["text"]
        prompt_text = prompt_dict["wiki"][idx]
        text1 = text[:len(prompt_text)-1]
        text2 = text[len(prompt_text):]  
        out = {"id":idx , "ended": ended, "length": length, "prompt_text": text1, "gen_text": text2}
        out_wiki.append(out)
        
with open(f"/root/autodl-tmp/gpt-2-output-dataset/pair/webtext.train_opt_{model}_top_50_wiki.sorted.split.{split}.pair.jsonl", "w") as file:
    print("Outputing paired jsonl file......")
    for out in tqdm(out_wiki):
        json.dump(out, file)
        file.write("\n")
