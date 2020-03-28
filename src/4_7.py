import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

from scipy.signal import freqz
from scipy.signal import butter, lfilter

wd = os.getcwd()
data_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_z_train.txt"
label_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/y_train.txt"

raw_data_df = pd.read_csv(data_file, sep="\s+", header=None)
label_df = pd.read_csv(label_file, sep="\s+", header=None)

raw_labeled_data = raw_data_df.assign(label=label_df.values)
raw_labeled_data = raw_labeled_data.drop(raw_labeled_data.columns[64:128], axis=1)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    cut = highcut / nyq
    b, a = butter(order, cut, btype='lowpass')
    return b, a

def highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    cut = lowcut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a

#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y

 # Sample rate and desired cutoff frequencies (in Hz).
fs = 50
lowcut = 3
highcut = 12
order = 4

original_signal = np.zeros(470528)
for index, row in raw_labeled_data.iterrows():
    eff = index * 64
    original_signal[eff:eff + 64] = row.values[:-1]

plt.figure(1)
plt.clf()

# Plot butter-band-filter.
b, a = butter_bandpass(lowcut, highcut, fs, order=order)
y = lfilter(b, a , original_signal)
plt.plot(y)
plt.xlabel('datapoints')
plt.title('Butter-band filter')
plt.grid(True)
plt.axis('tight')

plt.figure(2)
# Plot lowpass-filter.
b, a = lowpass(highcut, fs, order=order)
y = lfilter(b, a , original_signal)
plt.plot(y)
plt.xlabel('datapoints')
plt.title('Lowpass filter')
plt.grid(True)
plt.axis('tight')

plt.figure(3)
# Plot highpass-filter.
b, a = highpass(lowcut, fs, order=order)
y = lfilter(b, a , original_signal)
plt.plot(y)
plt.xlabel('datapoints')
plt.title('Highpass filter')
plt.grid(True)
plt.axis('tight')
plt.show()