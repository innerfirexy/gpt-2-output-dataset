from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

def _read_data(data_file, N=np.inf):
    data = []
    with open(data_file, 'r') as f:
        count = 0
        for line in f:
            line = line.strip()
            num = list(map(float, line.split()))
            data.append(num)
            count += 1
            if count >= N:
                break
    return data

def compute_freqs_powers(data):
    freqs, powers = [], []
    for i in tqdm.tqdm(range(len(data))):
        x = data[i]
        freq_x = fftshift(fftfreq(x.shape[-1]))
        sp_x = fftshift(fft(x))
        freq_x = freq_x[len(freq_x)//2:] # freq_x[freq_x >= 0]
        sp_x = sp_x[len(sp_x)//2:] # sp_x[freq_x >= 0]
        freqs.append(freq_x)
        powers.append(sp_x)
    return freqs, powers

def compute_freqs_powers_new(x):
    freq_x = fftshift(fftfreq(x.shape[-1]))
    sp_x = fftshift(fft(x))
    freq_x = freq_x[len(freq_x)//2:] # freq_x[freq_x >= 0]
    sp_x = sp_x[len(sp_x)//2:] # sp_x[freq_x >= 0]
    return freq_x, sp_x


def fp_pipeline(data_file, N=np.inf) -> pd.DataFrame:
    data_list = _read_data(data_file) # Read all data
    print(f"0406 debug len(data_list) is {len(data_list)}")
    data_arr = np.concatenate([np.asarray(d) for d in data_list])
    print(f"0406 debug data_arr shape {data_arr.shape}")
    mean_data = np.mean(data_arr)
    sd_data = np.std(data_arr)

    if N < np.inf:
        data_norm = [(np.asarray(d) - mean_data)/sd_data for d in data_list[:N]]
    else:
        data_norm = [(np.asarray(d) - mean_data)/sd_data for d in data_list]
    freqs, powers = compute_freqs_powers_new(data_arr)
    print(freqs[:100], powers[:100])
    df = pd.DataFrame.from_dict({
        'freq': freqs,
        'power': powers
    })
    return df

def nll2csv(source, split):
    input_path = f"../data/{source}.{split}.model=.nll"
    output_path = f'../plot/{source}_freq_power_fft_{split}.csv'
    df = fp_pipeline(input_path, N=5000) # read 5000 lines for each file
    df.to_csv(output_path, index=False)

sources = ['webtext', 'small-117M',  'small-117M-k40',
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40',
]

nll2csv('webtext', 'test')

# splits = ['train', 'valid', 'test']
# for source in sources:
#     for split in splits:
#         nll2csv(source, split)

# input_file = '../data/webtext.test.model=.nll'
# with open(input_file, 'r') as f:
#     line = f.readline()
#     print(f'input_file: {input_file} has {len(line)} lines')
# x = np.array(list(map(float, line.strip().split())))
# print(x.shape)

# # test line count
# with open(input_file, 'r') as f:
#     line_count = sum(1 for line in f)
#     print(f'input_file: {input_file} has {line_count} lines')

# freq_x = fftshift(fftfreq(x.shape[-1]))
# sp_x = fftshift(fft(x))
# print('freq_x:', freq_x.shape)
# print('sp_x:', sp_x.shape)

# # print value of freq_x in .2f format
# for i in range(freq_x.shape[0]):
#     print(f'{freq_x[i]:.2f}', end=' ')

# # print number of elements in freq_x > 0
# print('Method 1 length:', len(freq_x[freq_x > 0])) # 66
# print('Method 1 length:', len(freq_x[freq_x >= 0])) # 67

# # only keep the plot where x-axis value > 0
# plt.plot(freq_x[freq_x >= 0], sp_x.real[freq_x >= 0])
# plt.show()
# # plt.plot(freq_x, sp_x.real)

# # If we only plot 0 to 1/2 (right half)
# print('Method 2 length:', len(freq_x[len(freq_x)//2:]))
# plt.plot(freq_x[len(freq_x)//2:], sp_x[len(sp_x)//2:]) # 67
# plt.show()

# Concluions of comparison of Method 1 and Method 2: differ by one index

# # Test signal.periodogram()
# f, p = signal.periodogram(x)
# plt.plot(f, p)
# plt.show()

# # show min and max of freq_x
# print('min freq_x:', freq_x.min())
# print('max freq_x:', freq_x.max())