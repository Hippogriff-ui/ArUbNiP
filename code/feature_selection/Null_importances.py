import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import sys
import time

start = time.time()
data = pd.read_csv(sys.argv[1])
feature_df = data.iloc[:,1:]
var = np.var(feature_df,axis=0)
feature_name = var.index.tolist()
res = []
for i in range(len(var)):
    if not var[i]==0:
        res.append(i)
feature_var = [feature_name[i] for i in res]
feature_var.insert(0,'label')
df = data.loc[:,feature_var]

def get_feature_importances(data, shuffle, nb_runs):
    feat_name = data.columns[1:].tolist()
    X = data.values[:, 1:]
    
    y = data['label'].values.copy()
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    
    imp_df = pd.DataFrame({"feature": feat_name, "importance": feature_importances})
    imp_df_sorted = imp_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    if shuffle:
        null_imp_dfs = []
        for i in tqdm(range(nb_runs)):
            y = np.random.permutation(y)
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123)
            rf.fit(X, y)
            feature_importances = rf.feature_importances_
            null_imp_df = pd.DataFrame({"feature": feat_name, "importance": feature_importances})
            null_imp_df_sorted = null_imp_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
            null_imp_dfs.append(null_imp_df_sorted)
        null_imp_df = pd.concat(null_imp_dfs, keys=range(1, nb_runs + 1))
        imp_df_sorted = null_imp_df

    return imp_df_sorted
    
actual_imp_df = get_feature_importances(data=df, shuffle=False, nb_runs=0)
nonzero_actual_imp_df = actual_imp_df[actual_imp_df['importance'] > 0]

null_imp_df = get_feature_importances(data=df, shuffle=True, nb_runs=1000)
null_imp_df.to_csv(sys.argv[2], index=False)

print('+' * 50)

null_imp_mean = null_imp_df.groupby('feature')['importance'].mean().reset_index()

feature_scores = []
for f_ in tqdm(nonzero_actual_imp_df['feature']):
    f_actual_imps_gain = nonzero_actual_imp_df.loc[nonzero_actual_imp_df['feature'] == f_, 'importance'].values
    
    f_null_imp_mean = null_imp_mean.loc[null_imp_mean['feature'] == f_, 'importance'].values[0]

    final_scores = f_actual_imps_gain[0] / f_null_imp_mean
    feature_scores.append((f_, final_scores))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'final_score'])
sorted_df = scores_df.sort_values(by="final_score", ascending=False).reset_index(drop=True)

sorted_df.to_csv(sys.argv[3], index=False)

filtered_df = sorted_df.iloc[:round(sorted_df.shape[0]*0.05), :]
features_score_sorted = filtered_df['feature'].tolist()
features_score_sorted.insert(0,'label')
train_feat = data[features_score_sorted]
train_feat.to_csv(sys.argv[4], index=False)

end = time.time()
print('Running time: %s Seconds'%(end-start))
