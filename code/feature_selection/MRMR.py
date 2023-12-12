import numpy as np
import pandas as pd
from mrmr import mrmr_classif
import sys

train_data = pd.read_csv(sys.argv[1])

X = train_data.iloc[:,1:]
y = train_data.iloc[:,0]
K = int(sys.argv[2])
selected_features = mrmr_classif(X=X, y=y, K=K, n_jobs=-1)
selected_features.insert(0,'label')
selected_df = train_data[selected_features]
selected_df.to_csv(sys.argv[3], index=False)