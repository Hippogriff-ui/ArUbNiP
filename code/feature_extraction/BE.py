import numpy as np
import os
from tqdm import tqdm
import sys

def create_amino_acid_dict():

    A = np.eye(20)
    a = np.zeros((1, 20))
    A = np.vstack((A, a))

    B = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'V', 'Y', 'X'])

    amino_acid_dict = {}
    for k in range(21):
        amino_acid_dict[B[k]] = A[k, :]

    return amino_acid_dict
    
amino_acid_dict = create_amino_acid_dict()

def binary_code(input_path, output_dir):
    
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]

    row_num = len(stripped_lines)
    win_size = len(stripped_lines[0])

    binary_matrix = np.zeros((row_num, win_size*20))

    for i in range(row_num):
        seq = stripped_lines[i]
        binary_list = []
        for s in seq:
            binary_list.extend(amino_acid_dict[s])
        binary_matrix[i,:] = binary_list
        
    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_Binary'+'.txt',binary_matrix,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_Binary'+'.txt',binary_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_Binary'+'.txt',binary_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_Binary'+'.txt',binary_matrix,fmt='%g',delimiter=',')   
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        binary_code(input_path,output_dir)