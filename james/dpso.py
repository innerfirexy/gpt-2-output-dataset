from scipy.integrate import simpson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


freq = pd.read_csv(
    '/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/webtext.test.model=.csv')['freq'][0:67]
print(freq, type(freq), freq.shape)
freq = np.array(freq)


power = pd.read_csv(
    '/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/webtext.test.model=.csv')['power'][0:67]
print(power, type(power), power.shape)
power = np.array(power)

print(simpson(y=power, x=freq))


'''
GLM10b  2.179543780856077
Human 0.2685327897887675
'''
