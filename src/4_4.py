import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

wd = os.getcwd()
data_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_z_train.txt"
label_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/y_train.txt"

rawdata_df = pd.read_csv(data_file, sep = "\s+", header=None)
label_df = pd.read_csv(label_file, sep = "\s+", header=None)

raw_labeled_data = rawdata_df.assign(label=label_df.values)

#raw_labeled_data, error_check = raw_labeled_data.drop(raw_labeled_data.tail(1).index), raw_labeled_data.tail(1)
raw_labeled_data = raw_labeled_data.drop(raw_labeled_data.columns[64:128], axis=1)

#error_check = error_check.drop(error_check.columns[0:64], axis=1)
#error_check.columns = np.arange(0, 65)
#error_check = error_check.rename(columns={64: 'label'})

#raw_labeled_data = raw_labeled_data.append(error_check)

print(raw_labeled_data)

original_signal = np.array([])

for index, row in raw_labeled_data.iterrows():
    original_signal = np.concatenate((original_signal,row[:-1]), axis=None)

print(len(original_signal))
fig = plt.figure()
plt.plot(original_signal)
plt.show()