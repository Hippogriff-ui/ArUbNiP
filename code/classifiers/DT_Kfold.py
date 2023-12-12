import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import tree
from imblearn.over_sampling import SMOTE
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

def dt_kfold(X, y,  k_fold_cv):

    skf = StratifiedKFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    acc_dict = {}
    for m_ in tqdm(range(3, 11, 1)):
        key = 'max_depth = ' + str(m_)

        smote_dt_model = tree.DecisionTreeClassifier(max_depth = m_, random_state = 123)

        y_test_array = np.array([])
        y_pred_array = np.array([])

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            smote_x, smote_y = smote(X_train,y_train)

            smote_y_pred = smote_dt_model.fit(smote_x, smote_y).predict(X_test)

            y_test_array = np.concatenate((y_test_array, y_test))
            y_pred_array = np.concatenate((y_pred_array, smote_y_pred))

        acc_val = accuracy_score(y_test_array, y_pred_array)

        acc_dict[key] = acc_val
    
    max_key = max(acc_dict, key = acc_dict.get)
    max_value = acc_dict[max_key]
    
    return {max_key: max_value}
    
best_para = dt_kfold(X, y,  5)
print(best_para)