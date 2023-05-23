from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
from transformers import set_seed
from tqdm import tqdm

checkpoint = "/home/yyuan/bloomz_560m/"
model_name = 'bloomz_560m'

path = '/home/yyuan/'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             torch_dtype="auto",
                                             device_map="auto")
set_seed(32)
index = 1
filenames = ['truenews_35.txt', 'wikitext_35.txt']

for filename in filenames:
    with open(path + filename) as f:
        for line in tqdm(f.readlines()):
            if index % 3 == 0:
                inputs = tokenizer.encode(
                    'Expand the following sentence:' +
                    line.rsplit('\n')[0],
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt").to("cuda")
                outputs = model.generate(inputs,
                                         max_length=1024,
                                         do_sample=True)
                gen_text = tokenizer.decode(outputs[0], max_length=1024, skip_special_tokens= True)

                with jsonlines.open(
                        f"gen_{model_name}_{filename.split('.')[0]}.jsonl",
                        mode="a") as writer:
                    writer.write({
                        'prompt': line.rsplit('\n')[0],
                        'text': gen_text
                    })
            index += 1
