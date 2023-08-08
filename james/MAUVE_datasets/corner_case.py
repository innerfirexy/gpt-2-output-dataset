import pandas as pd
from tqdm import tqdm
import mauve


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

df = pd.read_csv('/Users/james/Workspace/gpt-2-output-dataset/james/human_mauve.csv')


count = 0
mauve_score = []
for index, row in df.iterrows():
    if row['Input.model_b'] == 'human' or row['Input.model_a'] == 'human':
        count += 1
        mauve_score.append(compute_mauve(row['Input.completionb'],row['Input.completiona'],1024))
        # print(mauve_score)
print(count)
dict = {'mauve': mauve_score}
df = pd.DataFrame(dict)
df.to_csv('corner_case.csv', index=False)