import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import sys
import time

def smote(X,y):
    smote = SMOTE(random_state = 123)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
    
def rf_kfold(X, y,  k_fold_cv):

    skf = StratifiedKFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    acc_dict = {}

    for n_ in tqdm(range(50, 151, 10)):
        key = 'n_estimators = ' + str(n_)
        
        smote_rf_model = RandomForestClassifier(n_estimators = n_, n_jobs = 4, random_state = 123)
        
        y_test_array = np.array([])
        y_pred_array = np.array([])

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            smote_x, smote_y = smote(X_train,y_train)

            smote_y_pred = smote_rf_model.fit(smote_x, smote_y).predict(X_test)

            y_test_array = np.concatenate((y_test_array, y_test))
            y_pred_array = np.concatenate((y_pred_array, smote_y_pred))
            
        acc_val = accuracy_score(y_test_array, y_pred_array)
            
        acc_dict[key] = acc_val
        
    max_key = max(acc_dict, key = acc_dict.get)
    max_value = acc_dict[max_key]
        
    return {max_key: max_value}
    
def rf_kfold_without_smote(X, y,  k_fold_cv):

    kf = KFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    acc_dict = {}

    for n_ in tqdm(range(50, 151, 10)):
        key = 'n_estimators = ' + str(n_)
        
        rf_model = RandomForestClassifier(n_estimators = n_, n_jobs = 4, random_state = 123)
        
        y_test_array = np.array([])
        y_pred_array = np.array([])

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_pred = rf_model.fit(X_train, y_train).predict(X_test)

            y_test_array = np.concatenate((y_test_array, y_test))
            y_pred_array = np.concatenate((y_pred_array, y_pred))
            
        acc_val = accuracy_score(y_test_array, y_pred_array)
            
        acc_dict[key] = acc_val
        
    max_key = max(acc_dict, key = acc_dict.get)
    max_value = acc_dict[max_key]
        
    return {max_key: max_value}

start = time.time()
data = pd.read_csv(sys.argv[1])
data_val = data.values
X = data_val[:,1:]
y = data_val[:,0]

best_para = rf_kfold(X, y, 5)
print(best_para)

end = time.time()
print('Running time: %s Seconds'%(end-start))
