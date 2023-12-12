import numpy as np
from Bio import SeqIO
import os
from tqdm import tqdm
import sys

def cksaap(input_path, output_dir, kmax):
    
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]
    
#     file = open(input_path, "r")
#     sequences = []
#     for seq_record in SeqIO.parse(file, "fasta"):
#         sequences.append(str(seq_record.seq))
#     file.close()

    # Define the amino acid alphabet
    aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    num_aa = len(aa_alphabet)
    rows = len(stripped_lines)
    feature_mat = np.zeros((rows, num_aa*num_aa*(kmax+1)))
    
    row_ind = 0
    for x in range(rows):
        s = stripped_lines[x]
        # Initialize the frequency matrix M
        M = np.zeros((num_aa, num_aa, kmax+1))
        
        # Count the number of occurrences of each amino acid pair
        for i in range(len(s)):
            for j in range(i+1, min(i+kmax+1, len(s))):
                if s[i] != 'X' and s[j] != 'X': 
                    aa_i = aa_alphabet.index(s[i])
                    aa_j = aa_alphabet.index(s[j])
                    k = j - i - 1
                    M[aa_i, aa_j, k] += 1

        # Normalize the frequency matrix to obtain the probability matrix P
        P = np.zeros((num_aa, num_aa, kmax+1))
        for k in range(kmax+1):
            norm = np.sum(M[:, :, k])
            if norm > 0:
                P[:, :, k] = M[:, :, k] / norm
            else:
                P[:, :, k] = 0

        # Flatten the probability matrix into a feature vector
        feat = P.reshape(num_aa*num_aa*(kmax+1),)
        feature_mat[row_ind,:] = feat
        row_ind += 1

    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_CKSAAP'+'.txt',feature_mat,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_CKSAAP'+'.txt',feature_mat,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_CKSAAP'+'.txt',feature_mat,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_CKSAAP'+'.txt',feature_mat,fmt='%g',delimiter=',')
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files= os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        cksaap(input_path,output_dir,4)