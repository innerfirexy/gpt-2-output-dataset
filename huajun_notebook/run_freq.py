from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import numpy as np
import pandas as pd
import tqdm
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
    for i in tqdm.tqdm(range(len(data))):
        f, p = signal.periodogram(data[i])
        freqs.append(f)
        powers.append(p)
    return freqs, powers


def compute_fft(data):
    freqs, powers = [], []
    for i in tqdm.tqdm(range(len(data))):
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


######
# About normalization:
# The following post suggest that we should normalize the input signal by dividing by the max.
# https://www.mathworks.com/matlabcentral/answers/356692-how-to-normalize-a-fft-to-plot-in-frequency-domain
######


def test():
    data_dir = '../data/gpt2-generated-from-prompt/split_new/'
    # output_dir = '../data/gpt2-generated-from-prompt/freq/'

    # FFT, not normalized
    #for input_file in input_files:
    for model_name in ['gpt2', 'gpt2-xl']:
        for domain in ['story_vary', 'truenews_35', 'wikitext_35']:
            for i in range(0, 1000, 200):
                input_file = f'{model_name}-{domain}.sorted.split.{i}.nll'
                df = fp_pipeline(data_dir + input_file, 'fft', normalize=False)
                output_file = data_dir + input_file[:-4] + '.fft.csv'
                df.to_csv(output_file, index=False)


if __name__ == '__main__':
    test()
    
