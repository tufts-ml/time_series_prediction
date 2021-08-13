#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio
import pandas as pd


# In[2]:

RAW_DATA_PATH = '/Users/mriley/Desktop/USC-HAD'


def remove_units(current_string):
    return current_string[:-2]


# In[11]:


# File 1: FeaturesPerSubject =====================================================================================

# Key(s): subject
# Features: age, height, weight


# In[36]:


# Returns single row of data:
#                            [subject, age, height, weight]
    
def get_one_row_per_subject(current_file):
    feat_per_sub_keys = ['subject','age','height','weight']
    for my_key in feat_per_sub_keys:
        if my_key == 'subject':
            sub_id = int(current_file[my_key][0])
        elif my_key == 'age':
            age = float(current_file[my_key][0])
            round(age,2)
        elif my_key == 'height':
            height = remove_units(current_file[my_key][0])
            height = float(height)
            round(height,2)
        else:
            weight = remove_units(current_file[my_key][0])
            weight = float(weight)
            round(weight,2)
            
    return sub_id,age,height,weight


# In[13]:


subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
activities = [1,2,3,4,5,6,7,8,9,10,11,12]
trials = [1,2,3,4,5]


# In[21]:


# Loop through subjects, produce csv file
# ---------------------------------------

per_subject_data = np.ones((14,4))
i=0
for sub in subjects:
    #os.path.join(RAW_DATA_PATH, str(sub), '/a1t1.mat')
    current_file = sio.loadmat(os.path.join(RAW_DATA_PATH, 'Subject'+str(sub), 'a1t1.mat'))
    #current_file = sio.loadmat('/Users/mriley/Desktop/USC-HAD/Subject'+str(sub)+'/a1t1.mat')
    sub_id,age,height,weight = get_one_row_per_subject(current_file)
    per_subject_data[i,:] = sub_id,age,height,weight
    i += 1

per_subject_df = pd.DataFrame(per_subject_data, columns = ['subject_id','age','height','weight'])

per_subject_df['subject_id'] = per_subject_df['subject_id'].astype(int)
per_subject_df['age'] = per_subject_df['age'].astype(int)
per_subject_df['height'] = per_subject_df['height'].astype(int)
per_subject_df['weight'] = per_subject_df['weight'].astype(int)

per_subject_df.to_csv('features_per_subject.csv',index=False)

print("--------------------------------------------")
print("features_per_subject.csv COMPLETE")
print("--------------------------------------------")

# In[22]:


# =====================================================================================
# ======================= End file 1: Features Per Subject ============================
# =====================================================================================


# In[29]:


# File 2: FeaturesPerTimestep ===================================================================================

# Key(s): subject, seq_id, timestep
# Features: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z


# In[30]:


def generate_uniq_seq_id(sub, tri, act):
    return 60*(sub-1)+tri+5*(act-1)


# In[31]:


subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
activities = [1,2,3,4,5,6,7,8,9,10,11,12]
trials = [1,2,3,4,5]


# In[26]:


# Loop through subjects, activities, trials, produce csv file
# -----------------------------------------------------------
temp_data = np.ones((2400,10))

for sub in subjects:
    for act in activities:
        for tr in trials:
            current_file = sio.loadmat(os.path.join(RAW_DATA_PATH, 'Subject'+str(sub), 'a'+str(act)+'t'+str(tr)+'.mat'))
            #current_file = sio.loadmat('/Users/mriley/Desktop/USC-HAD/Subject'+str(sub)+'/a'+str(act)+'t'+str(tr)+'.mat')
            
            current_sensor_data = current_file['sensor_readings']
            file_length = current_sensor_data.shape[0]
            
            # current subject_id & trial_no columns
            sub_id_col = sub*np.ones((file_length,1))
            trial_no_col = tr*np.ones((file_length,1))
            
            first_stack = np.hstack((sub_id_col,trial_no_col))

            # call function to get id
            seq_id = generate_uniq_seq_id(sub, tr, act)
            
            # add a column consisting of that id
            seq_id_col = seq_id*np.ones((file_length,1))
            
            second_stack = np.hstack((first_stack,seq_id_col))
            
            # add the timestep column
            tstep_col = np.arange(1,file_length+1)
            reshape_tsteps = tstep_col.reshape(((file_length,1)))
            
            third_stack = np.hstack((second_stack,reshape_tsteps))
            
            final_stack = np.hstack((third_stack,current_sensor_data))
            
            if seq_id == 1:
                temp_data = final_stack
            else:
                temp_data = np.vstack((temp_data,final_stack))

my_header="subject_id,trial,seq_id,tstep,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"

per_tstep_df = pd.DataFrame(temp_data, columns = ['subject_id','trial','seq_id','tstep','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'])

per_tstep_df['subject_id'] = per_tstep_df['subject_id'].astype(int)
per_tstep_df['trial'] = per_tstep_df['trial'].astype(int)
per_tstep_df['seq_id'] = per_tstep_df['seq_id'].astype(int)
per_tstep_df['tstep'] = per_tstep_df['tstep'].astype(int)

per_tstep_df.to_csv('features_per_timestep.csv',index=False)

print("--------------------------------------------")
print("features_per_timestep.csv COMPLETE")
print("--------------------------------------------")
# In[32]:


# =====================================================================================
# ======================= End file 2: Features Per Timestep ===========================
# =====================================================================================


# In[33]:


# File 3: OutcomesPerSequence ===================================================================================

# Key(s): subject, seq_id
# Outcomes: act_no, act_binary, act_three_classes


# In[34]:


def generate_uniq_seq_id(sub, tri, act):
    return 60*(sub-1)+tri+5*(act-1)


# In[37]:


# Returns single row of data:
#                             [sub_id, trial, uniq_id, act_no, act_binary, act_three_classes]

def generate_one_row_per_seq(current_file):
    feat_per_seq_keys = ['subject','trial','activity_number']
    
    for my_key in feat_per_seq_keys:
        if my_key == 'subject':
            sub_id = int(current_file[my_key][0])
        elif my_key == 'trial':
            trial = int(current_file[my_key][0])           
        else:
            try:
                act_no = int(current_file[my_key][0])
            except KeyError:
                act_no = int(current_file['activity_numbr'][0])
                
            if act_no < 6:
                act_three_classes = 1
                act_binary = 1
            elif act_no < 8:
                act_three_classes = 2
                act_binary = 1
            else:
                act_three_classes = 3
                act_binary = 0
                
    uniq_id = generate_uniq_seq_id(sub_id,trial,act_no)
    
    return sub_id, trial, uniq_id, act_no, act_binary, act_three_classes


# In[38]:


# Loop through subjects, activities, trials, produce csv file
# -----------------------------------------------------------

per_seq_outs = np.ones((840,6))
k=0
feat_per_seq_keys = ['subject','trial','activity_number']
for sub in subjects:
    for act in activities:
        for tr in trials:
            current_file = sio.loadmat(os.path.join(RAW_DATA_PATH, 'Subject'+str(sub), 'a'+str(act)+'t'+str(tr)+'.mat'))
            #current_file = sio.loadmat('/Users/mriley/Desktop/USC-HAD/Subject'+str(sub)+'/a'+str(act)+'t'+str(tr)+'.mat')    
            per_seq_outs[k,:] = generate_one_row_per_seq(current_file)
            k += 1     

outputs_df = pd.DataFrame(per_seq_outs, columns = ['subject_id','trial','seq_id','act_no','act_binary','act_three_classes'])

outputs_df['subject_id'] = outputs_df['subject_id'].astype(int)
outputs_df['trial'] = outputs_df['trial'].astype(int)
outputs_df['seq_id'] = outputs_df['seq_id'].astype(int)
outputs_df['act_no'] = outputs_df['act_no'].astype(int)
outputs_df['act_binary'] = outputs_df['act_binary'].astype(int)
outputs_df['act_three_classes'] = outputs_df['act_three_classes'].astype(int)

outputs_df.to_csv('outcomes_per_sequence.csv',index=False)


# In[39]:
print("--------------------------------------------")
print("outcomes_per_sequence.csv COMPLETE")
print("--------------------------------------------")

# =====================================================================================
# ======================= End file 3: Outcomes Per Sequence ===========================
# =====================================================================================


# In[ ]:





# In[ ]:




