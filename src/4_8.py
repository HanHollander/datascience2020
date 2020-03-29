import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

CLASSES = {1: 'WALKING',
           2: 'WALKING_UPSTAIRS',
           3: 'WALKING_DOWNSTAIRS',
           4: 'SITTING',
           5: 'STANDING',
           6: 'LAYING'}

wd = os.getcwd()
raw_data_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_z_train.txt"
feature_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/X_train.txt"
label_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/train/y_train.txt"
test_raw_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/test/Inertial_Signals/body_acc_z_test.txt"
test_feat_file =  wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/test/X_test.txt"
test_label_file = wd[:-4].replace(os.sep, "/") + "/data/UCI_HAR_Dataset/test/y_test.txt"

raw_X_train = pd.read_csv(raw_data_file, sep="\s+", header=None)
feat_X_train = pd.read_csv(feature_file, sep="\s+", header=None)
y_train = pd.read_csv(label_file, sep="\s+", header=None)

raw_X_test = pd.read_csv(test_raw_file, sep="\s+", header=None)
feat_X_test = pd.read_csv(test_feat_file, sep="\s+", header=None)
y_test = pd.read_csv(test_label_file, sep="\s+", header=None)

#raw_labeled_data = raw_data_df.assign(label=label_df.values)
#raw_labeled_data = raw_labeled_data.drop(raw_labeled_data.columns[64:128], axis=1)
raw_X_train = raw_X_train.drop(raw_X_train.columns[64:128], axis=1)
raw_X_test = raw_X_test.drop(raw_X_test.columns[64:128], axis=1)

raw_X_train = raw_X_train[:735]
feat_X_train = feat_X_train[:735]
y_train = y_train[:735]

raw_X_test = raw_X_test[:294]
feat_X_test = feat_X_test[:294]
y_test = y_test[:294]

#From https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python
def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1, metric=DTW)

#model.fit(raw_X_train, y_train.to_numpy().ravel())
model.fit(feat_X_train, y_train.to_numpy().ravel())
#label = model.predict(raw_X_test)
label = model.predict(feat_X_test)

print(classification_report(label, y_test.to_numpy().ravel(),
                            target_names=[l for l in CLASSES.values()]))

conf_mat = confusion_matrix(label, y_test.to_numpy().ravel())
print(conf_mat)