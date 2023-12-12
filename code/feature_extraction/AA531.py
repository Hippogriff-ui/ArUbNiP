import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys

aa531 = pd.read_excel("AA531properties.xlsx")

aa531_num = aa531.shape[1]-1
aa531_dict = {}
for i in range(aa531.shape[0]):
    aa531_dict[aa531.iloc[i,0]] = aa531.iloc[i,1:].values
    
def aa531_feature(input_path,output_dir):
    
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]
    
#     file = open(input_path, "r")
#     sequences = []
#     for seq_record in SeqIO.parse(file, "fasta"):
#         sequences.append(str(seq_record.seq))
#     file.close()
    
    rows = len(stripped_lines)
    win_size = len(stripped_lines[0])
    aa531_matrix = np.zeros((rows,win_size*aa531_num))

    for k in range(rows):
        seq = stripped_lines[k]
        list_531 = []
        for i in range(aa531_num):
            aa531_values = [aa531_dict[s][i] for s in seq]
            list_531.extend(aa531_values)
        aa531_matrix[k, :] = list_531
        
    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_AA531'+'.txt',aa531_matrix,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_AA531'+'.txt',aa531_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_AA531'+'.txt',aa531_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_AA531'+'.txt',aa531_matrix,fmt='%g',delimiter=',')
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        aa531_feature(input_path,output_dir)