import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

def smote(X,y):
    smote = SMOTE(random_state = 123)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
    
data = pd.read_csv(sys.argv[1])
data_val = data.values
X = data_val[:,1:]
y = data_val[:,0]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X = min_max_scaler.fit_transform(X)

def svc_kfold(X, y,  k_fold_cv):

    skf = StratifiedKFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    acc_dict = {}
    for c_ in tqdm([0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        for g_ in [1, 0.1, 0.01, 0.001, 0.0001]:
            key = 'C = ' + str(c_) + ',' + 'gamma = ' + str(g_)

            smote_svc_model = svm.SVC(C = c_, gamma = g_, random_state = 123)
            
            y_test_array = np.array([])
            y_pred_array = np.array([])

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                smote_x, smote_y = smote(X_train,y_train)

                smote_y_pred = smote_svc_model.fit(smote_x, smote_y).predict(X_test)

                y_test_array = np.concatenate((y_test_array, y_test))
                y_pred_array = np.concatenate((y_pred_array, smote_y_pred))

            acc_val = accuracy_score(y_test_array, y_pred_array)

            acc_dict[key] = acc_val
    
    max_key = max(acc_dict, key = acc_dict.get)
    max_value = acc_dict[max_key]
    
    return {max_key: max_value}
    
best_para = svc_kfold(X, y, 5)
print(best_para)
