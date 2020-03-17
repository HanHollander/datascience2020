import pandas as pd
import numpy as np
import os

wd = os.getcwd()
train_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/train/X_train.txt'
test_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/test/X_test.txt'
feature_file_path = wd[:-4] + '/data/UCI_HAR_Dataset/features.txt'
separator = '\s+'

with open(feature_file_path) as feature_file:
    names = feature_file.readlines()

train_df = pd.read_csv(train_file_path, sep=separator, header=0, names=names)
test_df = pd.read_csv(test_file_path, sep=separator, header=0, names=names)

print('a)')
print('')
print('train:')
print('\tcolumns (features): ' + str(len(train_df.keys())))
print('\trows (datapoints): ' + str(len(train_df.values)))
print('')
print('test:')
print('\tcolumns (features): ' + str(len(test_df.keys())))
print('\trows (datapoints): ' + str(len(test_df.values)))
print('')


print('b)')
print('')
tBodyAcc_entropy_X_train = train_df['23 tBodyAcc-entropy()-X\n']
train_mean = np.mean(tBodyAcc_entropy_X_train)
tBodyAcc_entropy_X_test = test_df['23 tBodyAcc-entropy()-X\n']
test_mean = np.mean(tBodyAcc_entropy_X_test)
print('mean 23 tBodyAcc-entropy()-X:')
print('\ttrain: ' + str(train_mean))
print('\ttest: ' + str(test_mean))
print('')
tBodyGyro_min_Z_train = train_df['135 tBodyGyro-min()-Z\n']
train_std = np.std(tBodyGyro_min_Z_train)
tBodyGyro_min_Z_test = test_df['135 tBodyGyro-min()-Z\n']
test_std = np.std(tBodyGyro_min_Z_test)
print('standard deviation 135 tBodyGyro-min()-Z:')
print('\ttrain: ' + str(train_std))
print('\ttest: ' + str(test_std))
print('')
tBodyAccMag_iqr_train = train_df['208 tBodyAccMag-iqr()\n']
train_median = np.median(tBodyAccMag_iqr_train)
tBodyAccMag_iqr_test = test_df['208 tBodyAccMag-iqr()\n']
test_median = np.median(tBodyAccMag_iqr_test)
print('median 208 tBodyAccMag-iqr():')
print('\ttrain: ' + str(train_median))
print('\ttest: ' + str(test_median))
print('')
fBodyAccJerk_bandsEnergy_25_32_train = train_df['399 fBodyAccJerk-bandsEnergy()-25,32\n']
train_lower = np.percentile(fBodyAccJerk_bandsEnergy_25_32_train, 25)
fBodyAccJerk_bandsEnergy_25_32_test = test_df['399 fBodyAccJerk-bandsEnergy()-25,32\n']
test_lower = np.percentile(fBodyAccJerk_bandsEnergy_25_32_test, 25)
print('lower percentile 399 fBodyAccJerk-bandsEnergy()-25,32:')
print('\ttrain: ' + str(train_lower))
print('\ttest: ' + str(test_lower))
print('')
fBodyAccMag_skewness_train = train_df['514 fBodyAccMag-skewness()\n']
train_upper = np.percentile(tBodyAcc_entropy_X_train, 75)
fBodyAccMag_skewness_test = test_df['514 fBodyAccMag-skewness()\n']
test_upper = np.percentile(tBodyAcc_entropy_X_test, 75)
print('upper percentile 514 fBodyAccMag-skewness():')
print('\ttrain: ' + str(train_upper))
print('\ttest: ' + str(test_upper))
print('')
