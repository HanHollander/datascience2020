import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

wd = os.getcwd()
train_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/train/y_train.txt'
test_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/test/y_test.txt'
img_path = wd[:-4] + '/img/4_2_b.png'
separator = '\s+'

train_df = pd.read_csv(train_file_path, sep=separator, header=0)
test_df = pd.read_csv(test_file_path, sep=separator, header=0)

print('a)')
print('')
print('\tcolumns (classification): 1')
print('\trows (datapoints): ' + str(len(train_df.values)))
print('')
print('b)')
print('')
unique_train, counts_train = np.unique(train_df.values, return_counts=True)
print('\tnumber of datapoints for each user activity in the training set: ' + str(dict(zip(unique_train, counts_train))))
unique_test, counts_test = np.unique(test_df.values, return_counts=True)
print('\tnumber of datapoints for each user activity in the testing set: ' + str(dict(zip(unique_test, counts_test))))
print('')
print('\tThe dataset is reasonably balanced (with class 2 and 3 having a little less datapoints, not significantly\n'
      '\tso). This means that each datapoint occurs approximately the same amount of times in the data, ensuring that\n'
      '\tthe training of the classifier happens for all of the classes equally (no class is trained poorly, leading\n'
      '\tto bad results if that class is represented more in the actual data).')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(unique_train - 0.167, counts_train, color='r', width=0.333, label='train')
ax.bar(unique_test + 0.166, counts_test, color='b', width=0.333, label='test')
ax.legend()
ax.title.set_text('4.2 b) Datapoints per activity')
ax.set_xlabel('Activity')
ax.set_ylabel('Datapoints')
plt.savefig(img_path)
