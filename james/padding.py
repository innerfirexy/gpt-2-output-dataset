import numpy as np
import pandas as pd
from itertools import zip_longest
from tqdm import tqdm

counter = 0
max_len = 1023
data = []
with open(
    "/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/5273_sample/webtext.train.model=.nll"
) as reader:
    for line in reader:
        line = line.strip()
        line = list(map(float, line.split()))
        # if len(line) > max_len:
        #     max_len = len(line)
        data.append(line)
        counter += 1
    # print(counter)
    print(max_len)  # 994
    
    padded_list = []
    for i in tqdm(range(len(data))):
        # print(len(d))
        if len(data[i]) < max_len:
            data[i].extend([9999999] * (max_len - len(data[i])))
        padded_list.append(data[i])
        if len(padded_list) != i + 1:
            raise ValueError("Error")
    print(len(padded_list))
    padded_data = np.array(padded_list)
    print(padded_data.shape)

with open("webtext.train.model.infpadded.nll", "w") as writer:
    for line in padded_data:
        res_str = " ".join(f"{num:.4f}" for num in line)
        writer.write(f"{res_str}\n")


# a = [[1,2,3],[1,2,3,4], [1,2,3,4,5]]
# for i in range(len(a)):
#     print(a[i])

# a = np.array(list(zip_longest(*a, fillvalue=0))).T
# print(a)
