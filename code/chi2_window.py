# python3.6

from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.motifs import create, Motif
from Bio.SeqUtils import ProtParam
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1], 'r') as f1:
    pos_lines = f1.readlines()
    
with open(sys.argv[2], 'r') as f2:
    neg_lines = f2.readlines()

my_alphabet = IUPAC.IUPACProtein.letters + "X"

pos_seqs = create([Seq(line.strip(),alphabet=my_alphabet) for line in pos_lines])
neg_seqs = create([Seq(line.strip(),alphabet=my_alphabet) for line in neg_lines])

pos_seqs_counts = pos_seqs.counts
neg_seqs_counts = neg_seqs.counts

pos_counts_df = pd.DataFrame(pos_seqs_counts)
neg_counts_df = pd.DataFrame(neg_seqs_counts)

p_list = []
chi2_list = []
position_num = pos_counts_df.shape[0]
for i in range(position_num):
    if i != 20:
        mat = np.zeros((2,20))
        mat[0,:] = pos_counts_df.iloc[i,:20]
        mat[1,:] = neg_counts_df.iloc[i,:20]

        chi2, p_value, dof, expected_freq = chi2_contingency(mat)
        p_list.append(p_value)
        chi2_list.append(chi2)
    else:
        p_list.append(np.nan)
        chi2_list.append(np.nan)
        
width = 0.5
cell = [i for i in range(-20, 21)]
index = np.arange(41)
threshold = -np.log10(0.05)
plt.figure(figsize=(6.5, 4))
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.bar(index, -np.log10(p_list), width, color="#352A86", edgecolor='black')
plt.axhline(y=threshold, color='#de425b', linestyle='-', label='-lg(0.05)',linewidth=1)
plt.xlabel('Position')
plt.ylabel('-lg(p-value)')
n = 2
plt.xticks(index[::n], cell[::n],fontsize = 10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(sys.argv[3], format='pdf', dpi=1200)
plt.show()