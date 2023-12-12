import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import sys

def smote(X,y):
    smote = SMOTE(random_state= 123)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
    
def xgb_kfold(X, y, output_path, k_fold_cv, n_estimators, subsample, gamma, method_name):

    output_path = output_path

    smote_result_dict = {}

    smote_result_dict['method'] = method_name

    skf = StratifiedKFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    smote_xgb_model = xgb.XGBClassifier(learning_rate = 0.01, 
                                                n_estimators = n_estimators, 
                                                gamma = gamma, 
                                                subsample = subsample,
                                                random_state = 123)


    y_test_array = np.array([])
    y_pred_array = np.array([])
    y_pred_proba_1 = np.array([])

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        smote_x, smote_y = smote(X_train,y_train)

        smote_xgb_model.fit(smote_x, smote_y)

        smote_y_pred = smote_xgb_model.fit(smote_x, smote_y).predict(X_test)
        
        smote_y_pred_proba_1 = smote_xgb_model.predict_proba(X_test)[:,1]
        
        y_test_array = np.concatenate((y_test_array, y_test))
        y_pred_array = np.concatenate((y_pred_array, smote_y_pred))
        y_pred_proba_1 = np.concatenate((y_pred_proba_1, smote_y_pred_proba_1))

    smote_result_dict['ACC'] = accuracy_score(y_test_array, y_pred_array)

    smote_result_dict['MCC'] = matthews_corrcoef(y_test_array, y_pred_array)
    
    smote_tn, smote_fp, smote_fn, smote_tp = confusion_matrix(y_test_array, y_pred_array).ravel()

    smote_result_dict['Sensitivity'] = smote_tp / (smote_tp + smote_fn)

    smote_result_dict['Specificity'] = smote_tn / (smote_tn + smote_fp)

    smote_result_dict['AUC'] = roc_auc_score(y_test_array, y_pred_proba_1)

    original_data = pd.read_excel(output_path)

    smote_result_df = pd.DataFrame(smote_result_dict, index=[0])

    smote_save_result = pd.concat([original_data, smote_result_df], axis=0)

    smote_save_result.to_excel(output_path, index=False)
    
def xgb_independent_test(X_train, y_train, X_test, y_test, output_path, n_estimators, subsample, gamma, method_name):

    output_path = output_path

    smote_result_dict = {}

    smote_result_dict['method'] = method_name

    smote_x, smote_y = smote(X_train,y_train)

    smote_xgb_model = xgb.XGBClassifier(learning_rate = 0.01, 
                                                n_estimators = n_estimators, 
                                                gamma = gamma, 
                                                subsample = subsample,
                                                random_state = 123)

    smote_xgb_model.fit(smote_x, smote_y)

    smote_y_pred = smote_xgb_model.predict(X_test)

    smote_accuracy = accuracy_score(y_test, smote_y_pred)

    smote_result_dict['ACC'] = smote_accuracy

    smote_mcc = matthews_corrcoef(y_test, smote_y_pred)

    smote_result_dict['MCC'] = smote_mcc

    smote_tn, smote_fp, smote_fn, smote_tp = confusion_matrix(y_test, smote_y_pred).ravel()

    smote_sensitivity = smote_tp / (smote_tp + smote_fn)

    smote_specificity = smote_tn / (smote_tn + smote_fp)

    smote_result_dict['Sensitivity'] = smote_sensitivity

    smote_result_dict['Specificity'] = smote_specificity

    smote_y_pred_proba = smote_xgb_model.predict_proba(X_test)

    smote_auc = roc_auc_score(y_test, smote_y_pred_proba[:,1])

    smote_result_dict['AUC'] = smote_auc

    original_data = pd.read_excel(output_path)

    smote_result_df = pd.DataFrame(smote_result_dict, index=[0])

    smote_save_result = pd.concat([original_data, smote_result_df], axis=0)

    smote_save_result.to_excel(output_path, index=False)
        
train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])
train_data = train_data.values
test_data = test_data.values
train_X = train_data[:,1:]
train_y = train_data[:,0]
test_X = test_data[:,1:]
test_y = test_data[:,0]

n_estimators = sys.argv[3]

subsample = sys.argv[4]

gamma = sys.argv[5]

method_name = sys.argv[6]

xgb_kfold(train_X, train_y, "kfold_result.xlsx", 5, n_estimators, subsample, gamma, method_name)

xgb_independent_test(train_X, train_y, test_X, test_y, "independent_test_result.xlsx", n_estimators, subsample, gamma, method_name)