import os

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from cycler import cycler

CLASSES = {1: 'WALKING',
           2: 'WALKING_UPSTAIRS',
           3: 'WALKING_DOWNSTAIRS',
           4: 'SITTING',
           5: 'STANDING',
           6: 'LAYING'}

wd = os.getcwd()
train_data_path = wd[:-4] + '/data/UCI_HAR_Dataset/train/X_train.txt'
test_data_path = wd[:-4] + '/data/UCI_HAR_Dataset/test/X_test.txt'
train_class_path = wd[:-4] + '/data/UCI_HAR_Dataset/train/y_train.txt'
test_class_path = wd[:-4] + '/data/UCI_HAR_Dataset/test/y_test.txt'
feature_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/features.txt'
separator = '\s+'

with open(feature_file_path) as feature_file:
    names = feature_file.readlines()

train_data_df = pd.read_csv(train_data_path, sep=separator, header=0, names=names)
test_data_df = pd.read_csv(test_data_path, sep=separator, header=0, names=names)
train_class_df = pd.read_csv(train_class_path, sep=separator, header=0)
test_class_df = pd.read_csv(test_class_path, sep=separator, header=0)

chosen_features = ['23 tBodyAcc-entropy()-X\n',
                   '24 tBodyAcc-entropy()-Y\n',
                   '135 tBodyGyro-min()-Z\n',
                   '136 tBodyGyro-sma()\n',
                   '208 tBodyAccMag-iqr()\n',
                   '209 tBodyAccMag-entropy()\n',
                   '399 fBodyAccJerk-bandsEnergy()-25,32\n',
                   '400 fBodyAccJerk-bandsEnergy()-33,40\n',
                   '514 fBodyAccMag-skewness()\n',
                   '515 fBodyAccMag-kurtosis()\n']

isolated = {}
for i in range(0, len(chosen_features)):
    feature = chosen_features[i]
    isolated[feature] = [[] for x in range(0, 6)]
    for j in range(0, len(train_class_df.values)):
        isolated[feature][(train_class_df.values[j][0] - 1)].append(train_data_df[feature][j])
    for j in range(0, len(test_class_df.values)):
        isolated[feature][(test_class_df.values[j][0] - 1)].append(test_data_df[feature][j])

for i in range(0, len(chosen_features)):
    feature = chosen_features[i]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']))
    for j in range(0, 6):
        sns.distplot(isolated[feature][j],
                     hist=False, rug=False, kde_kws={'shade': True}, label=CLASSES[j + 1], ax=ax)
    ax.legend()
    ax.title.set_text('4.3 Distribution of datapoint values per activity\nfeature: ' + feature[:-1] +
                      ' (train + test data)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Datapoints')
    plt.savefig(wd[:-4] + '/img/4_3_' +
                feature[:-1].replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '') +
                '.png')

print('There are often two classes, where the three walking activities are similar to each other, and the three non-\n'
      'activities too. This is easily explained, since in reality these activities also have a lot in common.')
print('')
print('23 tBodyAcc-entropy()-X')
print('')
print('\tWALKING:               normal distribution around 0.4, no skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around 0.35, no skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around 0.2, no skew, unimodal')
print('\tSITTING:               normal distribution around -0,7, slight right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,7, slight right skew, unimodal')
print('\tLAYING:                almost normal distribution around -0.6, right skew, almost unimodal')
print('\tCan discriminate between: WALKING/WALKING_UPSTAIRS/WALKING_DOWNSTAIRS, SITTING/STANDING, SLEEPING')
print('')
print('224 tBodyAcc-entropy()-Y')
print('')
print('\tWALKING:               normal distribution around 0.35, no skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around 0.25, slight left skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around 0.35, no skew, unimodal')
print('\tSITTING:               normal distribution around -0,65, right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,575, right skew, unimodal')
print('\tLAYING:                normal distribution around -0.7, right skew, unimodal')
print('\tCan discriminate between: WALKING/WALKING_UPSTAIRS/WALKING_DOWNSTAIRS, SITTING/STANDING/SLEEPING')
print('')
print('135 tBodyGyro-min()-Z')
print('')
print('\tWALKING:               normal distribution around 0.35, left skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around 0.35, no skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around 0.25, left skew, unimodal')
print('\tSITTING:               normal distribution around 0.8, slight left skew, unimodal')
print('\tSTANDING:              normal distribution around 0.8, slight left skew, unimodal')
print('\tLAYING:                normal distribution around 0.8, no skew, unimodal')
print('\tCan discriminate between: WALKING/WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING/STANDING/SLEEPING')
print('')
print('136 tBodyGyro-sma()')
print('')
print('\tWALKING:               normal distribution around -0.4, right skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around -0.25, right skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around -0.2, right skew, unimodal')
print('\tSITTING:               normal distribution around -0,95, slight right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,95, slight right skew, unimodal')
print('\tLAYING:                normal distribution around -0.95, slight right skew, unimodal')
print('\tCan discriminate between: WALKING, WALKING_UPSTAIRS/WALKING_DOWNSTAIRS, SITTING/STANDING/SLEEPING')
print('')
print('208 tBodyAccMag-iqr()')
print('')
print('\tWALKING:               almost normal distribution around -0.6, right skew, unimodal')
print('\tWALKING_UPSTAIRS:      flattened normal distribution around -0.5, slight right skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    flattened normal distribution around -0.25, no skew, unimodal')
print('\tSITTING:               normal distribution around -0,95, slight right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,95, slight right skew, unimodal')
print('\tLAYING:                normal distribution around -0.95, slight right skew, unimodal')
print('\tCan discriminate between: WALKING/WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING/STANDING/SLEEPING')
print('')
print('209 tBodyAccMag-entropy()')
print('')
print('\tWALKING:               normal distribution around 0.75, no skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around 0.85, no skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around 0.95, no skew, unimodal')
print('\tSITTING:               normal distribution around -0,75, right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,6, right skew, unimodal')
print('\tLAYING:                almost normal distribution around -0.8/-0.4, slight right skew, bimodal')
print('\tCan discriminate between: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING/STANDING, SLEEPING')
print('')
print('399 tBodyAccJerk-bandsEnergy()-25,32')
print('')
print('Difficult to discern any distribution information.')
print('')
print('400 tBodyAccJerk-bandsEnergy()-33,40')
print('')
print('Difficult to discern any distribution information.')
print('')
print('514 fBodyAccMag-skewness()')
print('')
print('\tWALKING:               normal distribution around -0.55, no skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around -0.5, no skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around -0.4, no skew, unimodal')
print('\tSITTING:               normal distribution around -0,65, right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,5, right skew, unimodal')
print('\tLAYING:                almost normal distribution around -0.7, right skew, unimodal')
print('\tCan discriminate between: WALKING/WALKING_DOWNSTAIRS, WALKING_UPSTAIRS, SITTING/SLEEPING, STANDING')
print('')
print('515 fBodyAccMag-kurtosis()')
print('')
print('\tWALKING:               normal distribution around -0.6, slight right skew, unimodal')
print('\tWALKING_UPSTAIRS:      normal distribution around -0.55, slight right right skew, unimodal')
print('\tWALKING_DOWNSTAIRS:    normal distribution around -0.6, slight right skew, unimodal')
print('\tSITTING:               normal distribution around -0,75, right skew, unimodal')
print('\tSTANDING:              normal distribution around -0,75, right skew, unimodal')
print('\tLAYING:                almost normal distribution around -0.75, slight right skew, bimodal')
print('\tCan discriminate between: WALKING/WALKING_UPSTAIRS/WALKING_DOWNSTAIRS/SITTING/STANDING/SLEEPING')
print('')

