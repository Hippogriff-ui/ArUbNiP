import numpy as np
import os
from tqdm import tqdm
import sys

def pwm(input_path, output_dir):
    
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]

    row_num = len(stripped_lines)
    win_size = len(stripped_lines[0])

    B = ['A', 'R', 'N', 'D', 'C', 'Q', 'E',
         'G', 'H', 'I', 'L', 'K', 'M', 'F',
         'P', 'S', 'T', 'W', 'V', 'Y', 'X']

    D = np.zeros((21, win_size))

    for i in range(21):
        for j in range(win_size):
            count = sum(1 for s in stripped_lines if s[j] == B[i])
            c = count / row_num
            D[i, j] = c

    pwm_matrix = np.zeros((row_num, win_size))

    for i in range(row_num):
        for j in range(win_size):
            k = B.index(stripped_lines[i][j])
            pwm_matrix[i, j] = D[k, j]

    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_PWM'+'.txt',pwm_matrix,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_PWM'+'.txt',pwm_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_PWM'+'.txt',pwm_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_PWM'+'.txt',pwm_matrix,fmt='%g',delimiter=',')
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        pwm(input_path,output_dir)