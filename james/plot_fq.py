import matplotlib.pyplot as plt
import numpy as np
import pandas

path = '/Users/james/Workspace/gpt-2-output-dataset/james/glm10b/5273_sample/'
data_zp = pandas.read_csv(path + 'webtext.test.model.zeropadded.csv')
# data_nonzp = pandas.read_csv('webtext.test.model.zeropadded.csv')

data_zp_power = np.array(data_zp['power']).reshape(5000, -1)
print(data_zp_power.shape)
data_zp_power_avg = np.mean(data_zp_power, axis=0)
plt.plot(data_zp['freq'][0:512],data_zp_power_avg)
plt.ylim(0, 2)

plt.show()
