
# coding: utf-8

# This script was used to generate the random item pairings. Numbers were taken in pairs (i.e. the first and second item in the sequence were paired, then the third and fourth, and so on) to construct item pairs. 14 of the original 54 items were set aside a priori to be used in the practice trials. These items were paired randomly as an independent set.

# In[39]:

import numpy as np

practice = [9, 11, 12, 14, 15, 24, 28, 32, 37, 39, 47, 52, 53, 54]
real = [x for x in range(1,55) if x not in practice]

practice_seq = np.random.permutation(practice).reshape(-1, 2)
real_seq = np.random.permutation(real).reshape(-1, 2)

print practice_seq
print real_seq


# The above script was run on October 7th, 2014, and produced the following sequences:
# 
# Practice (practice_seq):
# [[ 9 37]
#  [15 28]
#  [11 54]
#  [24 32]
#  [52 14]
#  [53 47]
#  [39 12]]
#  
# Real (real_seq):
# [[19 30]
#  [46 29]
#  [16 42]
#  [26 44]
#  [ 4 10]
#  [45 41]
#  [43 18]
#  [34 36]
#  [22  7]
#  [49 35]
#  [48 50]
#  [25 31]
#  [40  5]
#  [ 6 38]
#  [21  1]
#  [27 51]
#  [20  2]
#  [23 33]
#  [ 3 13]
#  [ 8 17]]
