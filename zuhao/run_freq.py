from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='', help='input file')
parser.add_argument('--output',
                    '-o',
                    type=str,
                    default='',
                    help='output file or dir')
parser.add_argument('--N', type=int, default=np.inf, help='number of samples')
parser.add_argument('--method',
                    type=str,
                    default='fft',
                    choices=['periodogram', 'fft'])


def _read_data(data_file, N=np.inf):
    data = []
    with open(data_file, 'r') as f:
        count = 0
        for line in f:
            line = line.strip()
            if line == '':
                continue
            num = list(map(float, line.split()))
            data.append(num)
            count += 1
            if count >= N:
                break
    return data


def compute_periodogram(data):
    freqs, powers = [], []
    for i in tqdm(range(len(data))):
        f, p = signal.periodogram(data[i])
        freqs.append(f)
        powers.append(p)
    return freqs, powers


def compute_fft(data):
    freqs, powers = [], []
    for i in tqdm(range(len(data))):
        x = data[i]
        try:
            freq_x = fftshift(fftfreq(x.shape[-1]))
            sp_x = fftshift(fft(x)).real
        except Exception:
            print(f'Error in sample {i}: {x}')
            raise
        freqs.append(freq_x[len(freq_x) // 2:])
        powers.append(sp_x[len(sp_x) // 2:])
    return freqs, powers


def fp_pipeline(data_file,
                method,
                n_samples=np.inf,
                normalize=False) -> pd.DataFrame:
    """
    :param data_file:
    :param method:
    :param n_samples:
    :param normalize: boolean, whether to normalize the data
    :return:
    """
    data_list = _read_data(data_file)  # Read all data
    data_arr = np.concatenate([np.asarray(d) for d in data_list])
    mean_data = np.mean(data_arr)
    sd_data = np.std(data_arr)

    if n_samples < np.inf:
        data = [np.asarray(d) for d in data_list[:n_samples]]
    else:
        data = [np.asarray(d) for d in data_list]
    if normalize:
        data = [(d - mean_data) / sd_data for d in data]

    if method == 'periodogram':
        freqs, powers = compute_periodogram(data)
    elif method == 'fft':
        freqs, powers = compute_fft(data)

    df = pd.DataFrame.from_dict({
        'freq': np.concatenate(freqs),
        'power': np.concatenate(powers)
    })
    return df


def extract_entropy(data_file):
    import json
    
    # Read all data
    entropy_list = []
    with open(data_file, "r", errors="replace") as file:
        for line in file:
            entropy = json.loads(line)["nll4tok"]
            entropy_list.append(entropy)       
        entropy_list.insert(0, data_file)
    return entropy_list


######
# About normalization:
# The following post suggest that we should normalize the input signal by dividing by the max.
# https://www.mathworks.com/matlabcentral/answers/356692-how-to-normalize-a-fft-to-plot-in-frequency-domain
######


def test():
    import os
    
    data_dir = "/root/autodl-tmp/gpt-2-output-dataset/data_degen/conditional/"
    input_files = os.listdir(data_dir)
    
    # Extract entropy
    entropy_all = []
    print("Extracting entropy......")
    for input_file in tqdm(input_files, total=len(input_files)):
        entropy_list = extract_entropy(data_dir + input_file)
        entropy_all.append(entropy_list)
        
    # Write entropy
    print("Writing entropy......")
    for entropy_list in tqdm(entropy_all, total=len(entropy_all)):
        output_file = entropy_list.pop(0).replace(".jsonl", ".nll")
        with open(output_file, "w") as file:
            for entropy in entropy_list:
                for value in entropy:
                    file.write(f"{value} ")
                file.write("\n")

    # FFT, not normalized
    input_files = [input_file for input_file in os.listdir(data_dir) if ".nll" in input_file]
    for input_file in tqdm(input_files, total=len(input_files)):
        df = fp_pipeline(data_dir + input_file, "fft", normalize=False)
        output_file = data_dir + input_file[:-4] + ".fft.csv"
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    test()
    
