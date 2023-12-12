import numpy as np
import os
from tqdm import tqdm
import sys

def eaac_encoding(input_path, output_dir, window_size):
    
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    num_amino_acids = len(amino_acids)
    sequence_length = len(stripped_lines[0])
    num_windows = sequence_length - window_size + 1
    rows = len(stripped_lines)
    
    output_matrix = np.zeros((rows, num_windows*num_amino_acids))
    for a in range(rows):
        seq = stripped_lines[a]
        encoding = []
        for i in range(num_windows):
            window = seq[i:i+window_size]
            for j, amino_acid in enumerate(amino_acids):
                count = window.count(amino_acid)
                encoding.append(count)
        output_matrix[a,:] = encoding
    
    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_EAAC'+'.txt',output_matrix,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_EAAC'+'.txt',output_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_EAAC'+'.txt',output_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_EAAC'+'.txt',output_matrix,fmt='%g',delimiter=',')
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files= os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        eaac_encoding(input_path, output_dir, 5)