from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn import preprocessing
import sys

def smote(X,y):
    smote = SMOTE(random_state=123)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
    
def para_convert(para):
    if '.' in para:
    return float(para)
else:
    return int(para)
    
train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

feature_selection_name = sys.argv[3]

svc_C = para_convert(sys.argv[4])
svc_gamma = para_convert(sys.argv[5])
max_depth = int(sys.argv[6])
rf_n_estimators = int(sys.argv[7])
xg_n_estimators = int(sys.argv[8])
xg_subsample = float(sys.argv[9])
xg_gamma = float(sys.argv[10])

train_data_val = train_data.values
test_data_val = test_data.values

train_X = train_data_val[:,1:]
train_y = train_data_val[:,0]

test_X = test_data_val[:,1:]
test_y = test_data_val[:,0]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
train_X_minmax = min_max_scaler.fit_transform(train_X)
test_X_minmax = min_max_scaler.transform(test_X)

# 5-fold roc curve
classifiers = {
    feature_selection_name+"+SVM": SVC(C = svc_C, gamma = svc_gamma, probability=True, random_state = 123),
    feature_selection_name+"+DT": DecisionTreeClassifier(max_depth = max_depth, random_state = 123),
    feature_selection_name+"+RF": RandomForestClassifier(n_estimators = rf_n_estimators, random_state = 123),
    feature_selection_name+"+XGBoost": xgb.XGBClassifier(learning_rate = 0.01, n_estimators = xg_n_estimators, subsample = xg_subsample, gamma = xg_gamma, random_state = 123)
}

skf = StratifiedKFold(n_splits = 5, random_state = 123, shuffle = True)

roc_data = {}

colors = {
    feature_selection_name+"+SVM": '#fbd279',
    feature_selection_name+"+DT": '#6ad6ac',
    feature_selection_name+"+RF": '#ef9a76',
    feature_selection_name+"+XGBoost": '#7a89aa'
}

plt.figure(figsize=(6, 5))

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12

for name, clf in classifiers.items():
    
    y_test_array = np.array([])
    y_pred_proba_array = np.array([])
    
    if 'SVM' not in name:
        for train_idx, test_idx in skf.split(train_X, train_y):
            X_train, X_test = train_X[train_idx], train_X[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            smote_x, smote_y = smote(X_train,y_train)
            clf.fit(smote_x, smote_y)
            y_score = clf.predict_proba(X_test)[:, 1]
            y_test_array = np.concatenate((y_test_array, y_test))
            y_pred_proba_array = np.concatenate((y_pred_proba_array, y_score))   
        
    else:
        for train_idx, test_idx in skf.split(train_X_minmax, train_y):
            X_train, X_test = train_X_minmax[train_idx], train_X_minmax[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            smote_x, smote_y = smote(X_train,y_train)
            clf.fit(smote_x, smote_y)
            y_score = clf.predict_proba(X_test)[:, 1]
            y_test_array = np.concatenate((y_test_array, y_test))
            y_pred_proba_array = np.concatenate((y_pred_proba_array, y_score))

    fpr, tpr, _ = roc_curve(y_test_array, y_pred_proba_array)
    
    auc_val = roc_auc_score(y_test_array, y_pred_proba_array)
    
    roc_data[name] = (fpr, tpr, auc_val)
    
    plt.plot(fpr, tpr, lw=1, label=f'{name} (AUC = {auc_val:.3f})', color=colors[name])

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(sys.argv[11], format='pdf', dpi=1200)

for name, (_, _, auc_val) in roc_data.items():
    print(f"{name}: AUC = {auc_val:.3f}")

plt.show()


# independent test roc curve
classifiers = {
    feature_selection_name+"+SVM": SVC(C = svc_C, gamma = svc_gamma, probability=True, random_state = 123),
    feature_selection_name+"+DT": DecisionTreeClassifier(max_depth = max_depth, random_state = 123),
    feature_selection_name+"+RF": RandomForestClassifier(n_estimators = rf_n_estimators, random_state = 123),
    feature_selection_name+"+XGBoost": xgb.XGBClassifier(learning_rate = 0.01, n_estimators = xg_n_estimators, subsample = xg_subsample, gamma = xg_gamma, random_state = 123)
}

colors = {
    feature_selection_name+"+SVM": '#fbd279',
    feature_selection_name+"+DT": '#6ad6ac',
    feature_selection_name+"+RF": '#ef9a76',
    feature_selection_name+"+XGBoost": '#7a89aa'
}

plt.figure(figsize=(6, 5))
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12
for name, clf in classifiers.items():
    if 'SVM' not in name:
        smote_train_X, smote_train_y = smote(train_X, train_y)
        clf.fit(smote_train_X, smote_train_y)
        y_score = clf.predict_proba(test_X)[:, 1]
    else:
        smote_train_X, smote_train_y = smote(train_X_minmax, train_y)
        clf.fit(smote_train_X, smote_train_y)
        y_score = clf.predict_proba(test_X_minmax)[:, 1]
    
    fpr, tpr, _ = roc_curve(test_y, y_score)
    auc_val = roc_auc_score(test_y, y_score)
    plt.plot(fpr, tpr, lw=1, label=f'{name} (AUC = {auc_val:.3f})', color=colors[name])

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.savefig(sys.argv[12], format='pdf', dpi=1200)

plt.show()
