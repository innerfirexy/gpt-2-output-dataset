from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import interpolate
from scipy.stats import pearsonr
from scipy.stats import spearmanr

 # # 从csv文件中读出每个区间，区分方法为每个区间必须升序
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

 # # 返回由散点模拟的函数（可以为线性，二次方程或者三次方程）
def getF(freq_list:list, power_list:list):
    f = interpolate.interp1d(freq_list, power_list, fill_value="extrapolate")
    return f

 # # 根据两个文件内容， 返回每个区间固定且相同间隔的x对应的y值，区间取值为[0, 0.5]
def alignPoints(filepath1:str, filepath2:str):

    freq_list_list_1, power_list_list_1 = getInterval(filepath1)
    freq_list_list_2, power_list_list_2 = getInterval(filepath2)
    print(
        f'There are {len(freq_list_list_1)},{len(freq_list_list_2)} intervals in file {filepath1},{filepath2} respectively')
    y1listlist, y2listlist = [], []

    short_length = len(freq_list_list_1) if len(freq_list_list_1)<len(freq_list_list_2) else len(freq_list_list_2)

    for i in range(short_length):
        freq_list1 = freq_list_list_1[i]
        power_list1 = power_list_list_1[i]
        freq_list2 = freq_list_list_2[i]
        power_list2 = power_list_list_2[i]

        func1 = getF(freq_list1, power_list1)
        func2 = getF(freq_list2, power_list2)

        # len1 = len(freq_list1)
        # len2 = len(freq_list2)

        # if len2 < len1:
        #     len1 = len2

        # interpolate
        x = np.linspace(0, 0.5, 1000)
        y1 = func1(x)
        y2 = func2(x)
        y1listlist.append(y1)
        y2listlist.append(y2)

    return x, y1listlist, y2listlist



 # # 为每个fre区间计算auc
# def getPSO(filepath1:str, filepath2:str):
#     area_floor_list, area_roof_list, pso_list = [], [], []

#     xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)

#     for i in range(len(y1listlist)):
#         y1list = y1listlist[i]
#         y2list = y2listlist[i]
#         ylists = []
#         ylists.append(y1list)
#         ylists.append(y2list)

#         y_intersection = np.amin(ylists, axis=0)
#         y_roof = np.amax(ylists, axis=0)
#         area_floor = np.trapz(y_intersection, xlist)
#         area_roof = np.trapz(y_roof, xlist)


#         area_floor_list.append(area_floor)
#         area_roof_list.append(area_roof)
#         pso_list.append(round(area_floor / area_roof, 4))

#     return area_floor_list, area_roof_list, pso_list

 # # 为每个fre区间计算auc
def getPSO(filepath1:str, filepath2:str):
    area_floor_list, area_roof_list, pso_list = [], [], []

    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]

        # # Check whether there is power value lower than 0. If so, move the whole spectrum upwards.
        # min1 = min(y1list)
        # min2 = min(y2list)
        # lowest_power = min(min1, min2)
        # if lowest_power<0:
        #     # print('Move the curve upwords for '+ str(lowest_power))
        #     y1list = [i - lowest_power for i in y1list]
        #     y2list = [i - lowest_power for i in y2list]
        y1list = [abs(i) for i in y1list]
        y2list = [abs(i) for i in y2list]
        ylists = []
        ylists.append(y1list)
        ylists.append(y2list)



        # plt.plot(x, y1, label=0)
        # plt.plot(x, y2, label=1)

        y_intersection = np.amin(ylists, axis=0)
        y_roof = np.amax(ylists, axis=0)
        area_floor = np.trapz(y_intersection, xlist)
        area_roof = np.trapz(y_roof, xlist)

        # fill_poly = plt.fill_between(x, 0, y_intersection, fc='yellow', ec='black', alpha=0.5,
        #                              label=f'intersection:{area_floor:.4f}')
        # fill_poly.set_hatch('xxx')
        # plt.legend()
        # plt.savefig('spectrum.png')

        area_floor_list.append(area_floor)
        area_roof_list.append(area_roof)
        pso_list.append(round(area_floor / area_roof, 4))

    return area_floor_list, area_roof_list, pso_list


def getSpearmanr(filepath1:str, filepath2:str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    corr_list = []

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]

        corr, _ = spearmanr(y1list, y2list)
        corr_list.append(corr)
    return corr_list

 # # 为每个fre区间计算PearsonCorelation
def getPearson(filepath1:str, filepath2:str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    corr_list = []

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]

        corr, _ = pearsonr(y1list, y2list)
        corr_list.append(corr)
    return corr_list

 # # Calculate the similarity between two spectra using Spectral Angle Mapper
def getSAM(filepath1:str, filepath2:str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    sam_list = []

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]
        ylists = []
        ylists.append(y1list)
        ylists.append(y2list)

        # Normalize the spectra
        y1list /= np.linalg.norm(y1list)
        y2list /= np.linalg.norm(y2list)

        # Calculate the dot product
        dot_product = np.dot(y1list, y2list)

        # Calculate the SAM similarity
        sam_similarity = np.arccos(dot_product) / np.pi

        sam_list.append(sam_similarity)

    return sam_list


# Following code is to calculate pso and other metrics between two single csv
# filepath1 = 'webtext_freq_power_1k_opt_125m_top_50_story_fft.csv'
# filepath2 = 'webtext_freq_power_1k_valid_fft.csv'
# area_floor_list, area_roof_list, pso_list = getPSO(filepath1, filepath2)
# corr_list = getPearson(filepath1, filepath2)
# sam_list = getSAM(filepath1, filepath2)
# spearmanr_list = getSpearmanr(filepath1, filepath2)


# Following code is to calculate pso and other metrics between many csvs
data_sources = ('news', 'story', 'wiki')
model_types = ('6.7b', '125m')
text_length_tuple = (0, 1, 2, 3, 4)
ans_str = ''

for model_type in model_types:
    for data_source in data_sources:
        total_pso_list = []
        total_corr_list = []
        total_sam_list = []
        total_spearmanr_list = []
        for text_length in text_length_tuple:
            original_filename = 'webtext.train.model=.' + data_source + '_' + str(text_length) + '.fft.csv'
            generated_filename = 'webtext.train_opt_' + model_type + '_top_50_' + data_source + '.sorted.split.' + str(text_length*200) + '.fft.csv'
            area_floor_list, area_roof_list, pso_list = getPSO(original_filename, generated_filename)
            corr_list = getPearson(original_filename, generated_filename)
            sam_list = getSAM(original_filename, generated_filename)
            spearmanr_list = getSpearmanr(original_filename, generated_filename)

            avg_pso = sum(pso_list) / len(pso_list)
            avg_corr = sum(corr_list) / len(corr_list)
            avg_sam = sum(sam_list) / len(sam_list)
            avg_spearmanr = sum(spearmanr_list) / len(spearmanr_list)
            tmp_str = model_type + '_' + data_source + '_' + str(text_length) + '\t' + str(avg_pso) + '\t' + str(avg_corr) + '\t' + str(
                avg_sam) + '\t' + str(avg_spearmanr) + '\n'
            ans_str = ans_str + tmp_str
            print(tmp_str)

            total_pso_list.extend(pso_list)
            total_corr_list.extend(corr_list)
            total_sam_list.extend(sam_list)
            total_spearmanr_list.extend(spearmanr_list)

        total_avg_pso = sum(total_pso_list)/len(total_pso_list)
        total_avg_corr = sum(total_corr_list)/len(total_corr_list)
        total_avg_sam = sum(total_sam_list)/len(total_sam_list)
        total_avg_spearmanr = sum(total_spearmanr_list)/len(total_spearmanr_list)

        tmp_str = model_type + '_' + data_source + '\t' + str(avg_pso) + '\t' + str(
            avg_corr) + '\t' + str(
            avg_sam) + '\t' + str(avg_spearmanr) + '\n'
        ans_str = ans_str + tmp_str
        print(tmp_str)

with open('Ans.txt','w') as f:
    f.write(ans_str)



# calculate the corelation between pso and other metrics.

# corr_pso_corr, _ = pearsonr(pso_list, corr_list)
# corr_pso_sam, _ = pearsonr(pso_list, sam_list)
# corr_pso_spear, _ = pearsonr(pso_list, spearmanr_list)
#
# print(f'The correlation are {0}, {1}, {2} between pso and pearson, pso and sam, pso and spearmanr'
#       , corr_pso_corr, corr_pso_sam, corr_pso_spear)