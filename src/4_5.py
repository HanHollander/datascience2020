import pandas as pd
import numpy as np
import scipy.stats as sst
import matplotlib.pylab as plt
import seaborn as sns
import os

from cycler import cycler

CLASSES = {1: 'WALKING',
           2: 'WALKING_UPSTAIRS',
           3: 'WALKING_DOWNSTAIRS',
           4: 'SITTING',
           5: 'STANDING',
           6: 'LAYING'}

wd = os.getcwd()
data_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_z_train.txt"
label_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/y_train.txt"

raw_data_df = pd.read_csv(data_file, sep="\s+", header=None)
label_df = pd.read_csv(label_file, sep="\s+", header=None)

raw_labeled_data = raw_data_df.assign(label=label_df.values)
raw_labeled_data = raw_labeled_data.drop(raw_labeled_data.columns[64:128], axis=1)

original_signal = np.zeros(470528)
for index, row in raw_labeled_data.iterrows():
    eff = index * 64
    original_signal[eff:eff + 64] = row.values[:-1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(original_signal)
ax.title.set_text('4.5 a) Original signal of body_acc_z_train')
ax.set_xlabel('Datapoint')
ax.set_ylabel('Value')
plt.savefig(wd[:-4] + '/img/4_5_full.png')

sizes = [1226, 1073, 986, 1286, 1373, 1407]
classified_signal = []
for size in sizes:
    classified_signal.append(np.zeros(size))
for i, row in raw_labeled_data.iterrows():
    classified_signal[int(row['label']) - 1] = \
        np.concatenate((classified_signal[int(row['label']) - 1], row.values[:-1]), axis=None)

fig = plt.figure()
fig.suptitle('4.5 a) Original signal of body_acc_z_train (classified)', size=16)

print('a)')
print('')
print('raw data used: body_acc_z_train.txt')
print('')
print('all classes:')
print('\tmean: ' + str(np.mean(original_signal)))
print('\tvariance: ' + str(np.var(original_signal)))
print('\tstd: ' + str(np.std(original_signal)))
print('\tskew: ' + str(sst.skew(original_signal)))
print('\tkurtosis: ' + str(sst.kurtosis(original_signal)))
print('\tmedian absolute deviation: ' + str(sst.median_absolute_deviation(original_signal)))
for i in range(0, 6):
    location = int('32' + str(i + 1))
    ax = fig.add_subplot(location)
    ax.plot(classified_signal[i])
    ax.title.set_text(CLASSES[i + 1])
    ax.set_xlabel('Datapoint')
    ax.set_ylabel('Value')
    print('class: ' + CLASSES[i + 1])
    print('\tmean: ' + str(np.mean(classified_signal[i])))
    print('\tvariance: ' + str(np.var(classified_signal[i])))
    print('\tstd: ' + str(np.std(classified_signal[i])))
    print('\tskew: ' + str(sst.skew(classified_signal[i])))
    print('\tkurtosis: ' + str(sst.kurtosis(classified_signal[i])))
    print('\tmedian absolute deviation: ' + str(sst.median_absolute_deviation(classified_signal[i])))

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(wd[:-4] + '/img/4_5_classified.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']))

for i in range(0, 6):
    dist = np.random.normal(np.mean(classified_signal[i]), np.std(classified_signal[i]), 20000)
    sns.distplot(dist,
                 hist=False, rug=False, kde_kws={'shade': True}, label=CLASSES[i + 1], ax=ax)

ax.legend()
ax.title.set_text('Generated normal distributions per class')
ax.set_xlabel('Value')
ax.set_ylabel('Datapoints')

plt.savefig(wd[:-4] + '/img/4_5_dists.png')

print('')
print('b)')
print('')
print('\tIt would be quite difficult to discriminate between the classes purely based on these distributions. If we\n'
      '\tlook at small differences we could discriminate all three walking activities separately (based on steepness\n'
      '\tand height of the distribution, but between the three stationary activities it would be very difficult. We\n'
      '\tcan see this in the plot in 4_5_dists.png.')
