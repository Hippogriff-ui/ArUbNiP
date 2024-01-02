import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import os
import sys

input_dir = sys.argv[1] # The training set and the test set need to combine features separately

pos_num = int(sys.argv[2])
neg_num = int(sys.argv[3])

feature_name = ['CKSAAP','PWM','EAAC','Binary','AA531', 'PSSM']

data_matrix = np.vstack((np.ones((pos_num,1)),np.zeros((neg_num,1))))

feat_name = ['label']

files= os.listdir(input_dir)

for f in tqdm(feature_name):
    single_feature = [_ for _ in files if f in _]
        
    if 'pos' in single_feature[0]:
        pos_file_name = single_feature[0]
        neg_file_name = single_feature[1]
    else:
        pos_file_name = single_feature[1]
        neg_file_name = single_feature[0]
        
    pos_data = np.loadtxt(input_dir+pos_file_name,delimiter=',')
    neg_data = np.loadtxt(input_dir+neg_file_name,delimiter=',')

    feature_col = [f+'_'+str(i) for i in range(1,pos_data.shape[1]+1)]
    feat_name.extend(feature_col)
    
    pos_neg = np.vstack((pos_data,neg_data))
    
    data_matrix = np.hstack((data_matrix,pos_neg))

df = pd.DataFrame(data_matrix, columns=feat_name)

df.to_csv(sys.argv[4], index=False)

print('Done!')
