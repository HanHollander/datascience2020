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
train_acc_x = wd[:-4] + '/data/UCI_HAR_Dataset/Inertial_Signals/train/boddy'
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