
# coding: utf-8

# # Libraries

# In[1]:


import os

import numpy as np
import pandas as pd
from math import floor, ceil

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import xarray as xr
#import tensorflow as tf

print('All packages imported.')


# # Data Import

# In[2]:


def get_file_path(file_name):
    CURRENT_DIR = os.getcwd()
    DATA_DIR = f'{CURRENT_DIR}/../../../data/stage-1_cleaned'
    FILE_PATH = f'{DATA_DIR}/{file_name}'
    return FILE_PATH


def import_DS(FILE_PATH):
    return xr.open_dataset(FILE_PATH)

def DS_dropna(DS):
    return DS.dropna(dim='time')


# In[3]:


# Get file
file_name = 'twparmbeatmC1.cdf'
FILE_PATH = get_file_path(file_name)

# Import data
armbeatm = import_DS(FILE_PATH)
armbeatm


# In[4]:


# Get file
file_name = 'twpqcrad1longC1.cdf'
FILE_PATH = get_file_path(file_name)

# Import data
qcrad = import_DS(FILE_PATH)
qcrad


# In[5]:


np.isnan(qcrad['down_short_diffuse_hemisp'].values).sum()


# In[6]:


np.isnan(qcrad['qc_down_short_diffuse_hemisp'].values).mean()


# # Data Filtering

# In[7]:


def np_filter(np_array, qc_array, good_class=0):
    if np_array.size != qc_array.size:
        raise Exception('Input arrays should have the same sizes.')
    else:
        return np.asarray([np_array[i] if qc_array[i] == good_class else np.nan for i in range(0,np_array.size)])


# In[8]:


# class 0 is good, class 1 is suspicious, class 2/3 are bad and missing
# Number of NaN and class 1
np.isnan(qcrad['down_short_diffuse_hemisp'].values).sum() + (qcrad['qc_down_short_diffuse_hemisp'].values == 1).sum()


# In[9]:


# After filtering, number of NaN should be the same as the number above (hurray!)
np_filtered = np_filter(qcrad['down_short_diffuse_hemisp'].values, 
                        qcrad['qc_down_short_diffuse_hemisp'].values)
print(np.isnan(np_filtered).sum())
np_filtered


# In[10]:


(qcrad['qc_down_short_diffuse_hemisp'].values == 2).sum()


# In[11]:


qcrad['down_short_diffuse_hemisp'].values[qcrad['qc_down_short_diffuse_hemisp'].values == 1].sum()


# In[12]:


da = xr.DataArray(np_filtered, coords=[('time', qcrad.time)], name='down_short_diffuse_hemisp')
da


# # Merge

# In[ ]:


DS = xr.merge([armbeatm, da]).dropna(dim='time')
DS


# In[1]:


def save_netcdf(DS, FILE_PATH):
    DS.to_netcdf(FILE_PATH)
    print('Saved.')
    return None


def get_save_file_path(file_name, stage=1):
    CURRENT_DIR = os.getcwd()
    DATA_DIR = f'{CURRENT_DIR}/../../../data/stage-{stage}_cleaned'
    FILE_PATH = f'{DATA_DIR}/{file_name}'
    return FILE_PATH


# In[ ]:


# Save file
file_name = 'merged_dropped.cdf'
FILE_PATH = get_save_file_path(file_name)

save_netcdf(DS, FILE_PATH)

