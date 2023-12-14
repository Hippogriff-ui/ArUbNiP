import numpy as np
import pandas as pd
import sys

test_combine_data = pd.read_csv(sys.argv[1])

train_data = pd.read_csv(sys.argv[2])

train_col = train_data.columns.tolist()
df = test_data[train_col]
df.to_csv(sys.argv[3], index=False)
