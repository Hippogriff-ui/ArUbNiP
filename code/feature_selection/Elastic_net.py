from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
import sys

train_data = pd.read_csv(sys.argv[1])

def elastic_net(X, y, feature_names):
    enet = ElasticNet(random_state=123)
    enet.fit(X, y)
    w_nonzero = enet.coef_ != 0
    selected_features = feature_names[w_nonzero]
    return selected_features
    
feature_names = train_data.columns.values[1:]
train_val = train_data.values
X = train_val[:,1:]
y = train_val[:,0]

selected_features = elastic_net(X, y, feature_names)

selected_features = list(selected_features)

selected_features.insert(0,'label')
selected_df = train_data[selected_features]
selected_df.to_csv(sys.argv[2], index=False)