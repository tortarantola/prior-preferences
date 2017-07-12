
# coding: utf-8

# In[4]:

# This script was used to generate the random item choices that participants were asked to infer during the practice block of the inference task.
import numpy as np

practice_pairs = [[9, 37], [15, 28], [11, 54], [24, 32], [52, 14], [53, 47], [39, 12]]
practice_pairs_choices = []
for x in practice_pairs:
    choice = np.random.choice(x, size=None)
    practice_pairs_choices.append(choice)
print practice_pairs_choices



# The above script was run on October 8th, 2014, and produced the following choices:
# [37, 15, 11, 32, 52, 53, 12]
