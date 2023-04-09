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
parser.add_argument('--method', type=str, default='fft', choices=['periodogram', 'fft'])


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
        freqs.append(freq_x[len(freq_x)//2:])
        powers.append(sp_x[len(sp_x)//2:])
    return freqs, powers


def fp_pipeline(data_file, method, n_samples=np.inf, normalize=False) -> pd.DataFrame:
    data_list = _read_data(data_file) # Read all data
    data_arr = np.concatenate([np.asarray(d) for d in data_list])
    mean_data = np.mean(data_arr)
    sd_data = np.std(data_arr)

    if n_samples < np.inf:
        data = [np.asarray(d) for d in data_list[:n_samples]]
    else:
        data = [np.asarray(d) for d in data_list]
    if normalize:
        data = [(d - mean_data)/sd_data for d in data]

    if method == 'periodogram':
        freqs, powers = compute_periodogram(data)
    elif method == 'fft':
        freqs, powers = compute_fft(data)

    df = pd.DataFrame.from_dict({
        'freq': np.concatenate(freqs),
        'power': np.concatenate(powers)
    })

    return df


def test():
    data_dir = '../data/data_gpt2_old/'
    # input_files = ['small-117M.test.model=gpt2.nll',
    #                'small-117M.test.model=gpt2-medium.nll',
    #                'small-117M.test.model=gpt2-large.nll',
    #                'small-117M.test.model=gpt2-xl.nll']
    input_files = ['webtext.test.model=gpt2.nll',
                   'webtext.test.model=gpt2-medium.nll',
                   'webtext.test.model=gpt2-large.nll',
                   'webtext.test.model=gpt2-xl.nll']

    # Periodogram, normalized
    for input_file in input_files:
        df = fp_pipeline(data_dir + input_file, 'periodogram', normalize=True)
        output_file = data_dir + input_file[:-4] + '.periodogram.normalized.csv'
        df.to_csv(output_file, index=False)

    # Periodogram, not normalized
    for input_file in input_files:
        df = fp_pipeline(data_dir + input_file, 'periodogram', normalize=False)
        output_file = data_dir + input_file[:-4] + '.periodogram.csv'
        df.to_csv(output_file, index=False)

    # FFT, normalized
    for input_file in input_files:
        df = fp_pipeline(data_dir + input_file, 'fft', normalize=True)
        output_file = data_dir + input_file[:-4] + '.fft.normalized.csv'
        df.to_csv(output_file, index=False)

    # FFT, not normalized
    for input_file in input_files:
        df = fp_pipeline(data_dir + input_file, 'fft', normalize=False)
        output_file = data_dir + input_file[:-4] + '.fft.csv'
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    test()