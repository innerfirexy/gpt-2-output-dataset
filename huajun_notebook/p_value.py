import json
import numpy as np
from tqdm import tqdm
import mauve
from transformers import AutoTokenizer
from simctg.evaluation import measure_repetition_and_diversity
# from simcse import SimCSE
from Bleu import Bleu
from SelfBleu import SelfBleu
import pandas as pd

import os

from matplotlib import pyplot as plt

from pandas import DataFrame
from scipy import interpolate
from scipy.stats import ttest_ind

def getInterval(fre_power_filepath:str):
    spectrum = pd.read_csv(fre_power_filepath)
    spectrum['group'] = (spectrum['freq'].shift(1) > spectrum['freq']).cumsum()
    grouped_spectrum = spectrum.groupby('group')
    freq_list = []
    power_list = []

    for name, group in grouped_spectrum:
        freq_list.append(group['freq'].tolist())
        power_list.append(group['power'].tolist())

    return freq_list, power_list


def getF(freq_list:list, power_list:list):
    f = interpolate.interp1d(freq_list, power_list)
    return f

 # # 为每个fre区间计算auc
def getPSO(filepath1:str, filepath2:str):
    freq_list_list_1, power_list_list_1 = getInterval(filepath1)
    freq_list_list_2, power_list_list_2 = getInterval(filepath2)
    print(f'There are {len(freq_list_list_1)},{len(freq_list_list_2)} intervals in file {filepath1},{filepath2} respectively')

    area_floor_list, area_roof_list, pso_list = [], [], []

    for i in range(len(freq_list_list_1)):
        freq_list1 = freq_list_list_1[i]
        power_list1 = power_list_list_1[i]
        freq_list2 = freq_list_list_2[i]
        power_list2 = power_list_list_2[i]

        func1 = getF(freq_list1, power_list1)
        func2 = getF(freq_list2, power_list2)

        # interpolate
        x = np.linspace(0, 0.45, 200)
        y1 = func1(x)
        y2 = func2(x)

        ys = []
        ys.append(y1)
        ys.append(y2)

        # plt.plot(x, y1, label=0)
        # plt.plot(x, y2, label=1)

        y_intersection = np.amin(ys, axis=0)
        y_roof = np.amax(ys, axis=0)
        area_floor = np.trapz(y_intersection, x)
        area_roof = np.trapz(y_roof, x)

        # fill_poly = plt.fill_between(x, 0, y_intersection, fc='yellow', ec='black', alpha=0.5,
        #                              label=f'intersection:{area_floor:.4f}')
        # fill_poly.set_hatch('xxx')
        # plt.legend()
        # plt.savefig('spectrum.png')

        area_floor_list.append(area_floor)
        area_roof_list.append(area_roof)
        pso_list.append(round(area_floor / area_roof, 4))

    return area_floor_list, area_roof_list, pso_list

def getPSO_mean(filepath1:str, filepath2:str):
    freq_list_list_1, power_list_list_1 = getInterval(filepath1)
    freq_list_list_2, power_list_list_2 = getInterval(filepath2)
    print(f'There are {len(freq_list_list_1)},{len(freq_list_list_2)} intervals in file {filepath1},{filepath2} respectively')

    area_floor_list, area_roof_list, pso_list = [], [], []

    for i in range(len(freq_list_list_1)):
        freq_list1 = freq_list_list_1[i]
        power_list1 = power_list_list_1[i]
        freq_list2 = freq_list_list_2[i]
        power_list2 = power_list_list_2[i]

        func1 = getF(freq_list1, power_list1)
        func2 = getF(freq_list2, power_list2)

        # interpolate
        x = np.linspace(0, 0.45, 200)
        y1 = func1(x)
        y2 = func2(x)

        ys = []
        ys.append(y1)
        ys.append(y2)

        # plt.plot(x, y1, label=0)
        # plt.plot(x, y2, label=1)

        y_intersection = np.amin(ys, axis=0)
        y_roof = np.amax(ys, axis=0)
        area_floor = np.trapz(y_intersection, x)
        area_roof = np.trapz(y_roof, x)

        # fill_poly = plt.fill_between(x, 0, y_intersection, fc='yellow', ec='black', alpha=0.5,
        #                              label=f'intersection:{area_floor:.4f}')
        # fill_poly.set_hatch('xxx')
        # plt.legend()
        # plt.savefig('spectrum.png')

        area_floor_list.append(area_floor)
        area_roof_list.append(area_roof)
        pso_list.append(round(area_floor / area_roof, 4))

    return np.mean(pso_list)

# getPSO_mean('plot/small-117M_freq_power_1k_train.csv', 'plot/webtext_freq_power_1k_train.csv')
sources = ['webtext', 'small-117M',  'small-117M-k40',
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40',
]
# splits = ['train', 'valid', 'test']
split = 'test'
res_2D = list()
res_2D.append(sources)
df = pd.DataFrame(columns=['col'] + sources)
for source in sources:
    res_1D = list()
    res_1D.append(source)
    for source2 in sources:
    # for split in splits:
        try:
            _, _, v1 = getPSO(f'plot/{source}_freq_power_1k_{split}.csv', f'plot/webtext_freq_power_1k_{split}.csv')
            _, _, v2 = getPSO(f'plot/{source2}_freq_power_1k_{split}.csv', f'plot/webtext_freq_power_1k_{split}.csv')
            # "{:.2f}".format(z)
            res_1D.append("{:.2f}".format(ttest_ind(v1, v2).pvalue))
        except:
            res_1D.append('value error')
    df.loc[len(df)] = res_1D.copy()
    # res_2D.append(res_1D.copy())

#df = pd.DataFrame(np.array(res_2D))
df.to_csv('data/metrics/p-value.csv', index=False)

# df = pd.DataFrame(columns=['model_name', 'PSO', 'rep-2', 'rep-3', 'rep-4', 'diversity', 'bleu', 'self-bleu', 'mauve', 'coherence'])
# df.loc[len(df)] = [model_name, 'PSO', rep_2, rep_3, rep_4, div_score, bleu_score, self_bleu_score, mauve_score,
#                    coherence]
# df.to_csv(f"data/metrics/{model_name}_metrics.csv", index=False)