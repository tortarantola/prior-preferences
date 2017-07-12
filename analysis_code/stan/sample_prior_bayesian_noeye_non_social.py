##################################################
### DRIFT DIFFUSION MODEL: BAYESIAN w/BIASED PRIOR (NON-SOCIAL GROUP) ###
##################################################


# Import necessary libraries
import pandas as pd
import numpy as np
import pystan
import pickle

# Import data
data = pd.read_pickle(r'../../../../data/control/data_processed_control_wexclusions.pkl')

data2 = data[['participant','img_correct','response_correct','infer_resp_rt','feedback_correct','inf_bid_dv','block_loop_thisN']].copy()
data2 = data2[data2['block_loop_thisN'].notnull() & data2['img_correct'].notnull()]

model_data_rt = []
for p in data2['participant'].unique():
    pair_list_rt = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = list(data2[(data2['participant']==p) & (data2['img_correct']==i)].infer_resp_rt)
        pair_list_rt.append(x)
    model_data_rt.append(pair_list_rt)
missing = np.where(np.isnan(model_data_rt)==True)
for x in range(0,len(missing[0])):
    model_data_rt[missing[0][x]][missing[1][x]][missing[2][x]] = -1 # Replace missing response time data with -1, which we'll identify and exclude in the Stan model

model_data_correct = []
for p in data2['participant'].unique():
    pair_list_correct = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = map(int,list(data2[(data2['participant']==p) & (data2['img_correct']==i)].response_correct))
        pair_list_correct.append(x)
    model_data_correct.append(pair_list_correct)

model_data_feedback = []
for p in data2['participant'].unique():
    pair_list_feedback = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = map(int,list(data2[(data2['participant']==p) & (data2['img_correct']==i)].feedback_correct))
        pair_list_feedback.append(x)
    model_data_feedback.append(pair_list_feedback)

model_data_bid_congruence = [] # Bid for correct item minus bid for incorrect item
for p in data2['participant'].unique():
    pair_list_bid_congruence = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = list(data2[(data2['participant']==p) & (data2['img_correct']==i)].inf_bid_dv)
        pair_list_bid_congruence.append(x)
    model_data_bid_congruence.append(pair_list_bid_congruence)

model_rt_mins = [] # Create a list of each subject's lowest response time
for p in data.participant.unique():
    model_rt_mins.append(data[data['participant']==p].infer_resp_rt.min())
model_rt_mins[3] = 0.41 # Replace subject C12, C26, and C31's implausibly low minimum RT with the lowest RT from all other subjects' data
model_rt_mins[14] = 0.41 
model_rt_mins[20] = 0.41 

model_data = {'NS': 30, 'NP': 20, 'NT': 30, 'correct': model_data_correct, 'feedback': model_data_feedback, 'bid_congruence': model_data_bid_congruence, 'rt': model_data_rt, 'rt_mins': model_rt_mins}

model_code_obj = pystan.StanModel(file='model_prior_bayesian_noeye_non_social.stan.cpp', model_name='model_prior_bayesian_noeye_non_social') # Specific to model
fit = model_code_obj.sampling(data=model_data, iter=2000, chains=4, refresh=10)

with open('pickles/model_prior_bayesian_noeye_non_social.pkl', 'wb') as f: # Specific to model
    pickle.dump(model_code_obj, f)

with open('pickles/fit_prior_bayesian_noeye_non_social.pkl', 'wb') as f: # Specific to model
    pickle.dump(fit, f)

print(fit)