
# coding: utf-8

# In[ ]:


# Install xarray using conda
#!pip install xarray
#!pip install netcdf4
get_ipython().system(u'wget -c https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh')
get_ipython().system(u'chmod +x Anaconda3-5.1.0-Linux-x86_64.sh')
get_ipython().system(u'bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system(u'conda install -q -y --prefix /usr/local -c conda-forge xarray dask netCDF4 bottleneck')

import sys
sys.path.append('/usr/local/lib/python3.6/site-packages/')


# In[ ]:


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, ceil
import seaborn as sns

import xarray as xr
#import tensorflow as tf
print('All packages imported.')


# In[13]:


# Mount Google Drive locally
from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


# Check data list
get_ipython().system(u'ls "/content/gdrive/My Drive/Colab Notebooks/data/"')


# In[14]:


# Read data
f = "/content/gdrive/My Drive/Colab Notebooks/data/ARM_no_dropna.cdf"
DS = xr.open_dataset(f)
DS = DS.drop(['z10','z2'])
DS


# In[15]:


# Build multiple DataArray to collapse pressure coord
precip = DS['prec_sfc'] # the only DataArray with no pressure coord
binwidth = 5 # 5mm range for each bin of histogram
plt.hist(precip.dropna(dim='time'), bins=range(20, 100, binwidth)) # ignore rains with <20mm
plt.show()

da_list = ['precip'] # initialize list of DataArray

plev = DS['p'].values.astype(np.int32) # array of pressure level
var_p_list = ['T_p','rh_p','u_p','v_p'] # array of variables with pressure coord

for _var in var_p_list:
  for _p in plev:
    new_var_str = (_var + str(_p)) # new variable name (in string)
    da_list.append(new_var_str) # append the new variable
    exec("{} = DS[_var].sel(p=_p).drop('p')".format(new_var_str)) # assign DataArray
    exec("{} = {}.rename('{}')".format(new_var_str,new_var_str,new_var_str)) # rename data variables to avoid merge collision


# In[16]:


# Merge DataArray and Build a new DataSet
DS_new = xr.Dataset()
for _var in da_list:
  exec("DS_new = xr.merge([DS_new,{}])".format(_var))
  
DS_new


# In[17]:


# Convert DataSet to dataframe
df = DS_new.to_dataframe().reset_index() # convert index (time) to a column

# Find time difference between consecutive entries
hour_delta = df['time'].astype(np.int64).rolling(window=2).apply(lambda j: j[1] - j[0]) / 10**9 / 3600
for _hd in range(1,24+1):
  _cnt = hour_delta[hour_delta==_hd].count()
  if _cnt > 0:
    print('hour_delta= {}, count= {}'.format(_hd,_cnt))


# In[ ]:


# Find out the shift of the parameter (column) of interest that maximizes our non-NaN sample size

for shift in range(1,24+1):
  df_shift = df.copy() # deep copy
  prec_sfc_next = df['prec_sfc'][shift:].values # turn to np values to get rid of index
  for j in range(1,shift+1):
    prec_sfc_next = np.append(prec_sfc_next,float('NaN')) # consistent length
  df_shift.insert(loc=0, column='prec_sfc_next', value=prec_sfc_next) # insert the shifted column
  print('#shifted hour= ' + str(shift) + ', sample size= ' + str((~df_shift.isnull().T.any()).sum()))


# In[26]:


# Decided to take 1-hour shifted precipitation as labels

prec_sfc_1hrlater = df['prec_sfc'][1:].values # ditto
prec_sfc_1hrlater = np.append(prec_sfc_1hrlater,float('NaN')) 
try: # enable re-run
  df.insert(loc=0, column='prec_sfc_1hrlater', value=prec_sfc_1hrlater)
except:
  print('Column prec_sfc_1hrlater has already inserted.')

df_no_nan = df[~df.isnull().T.any()] # delete rows with incomplete data
print('#shifted hour= ' + str(1) + ', sample size= ' + str(len(df_no_nan)) + ', it should match the previous')
df_no_nan


# In[28]:


# Generate cyclic hour
hour24 = df_no_nan['time'].dt.hour.values
theta_hour = 2*np.pi * hour24/24
try:
  df_no_nan.insert(loc=0, column='hour_sin', value=np.sin(theta_hour))
except:
  print('Column hour_sin has already inserted.')
try:
  df_no_nan.insert(loc=0, column='hour_cos', value=np.cos(theta_hour))
except:
  print('Column hour_cos has already inserted.')
  
df_nn = df_no_nan.reset_index().drop(columns=['time','index']) # delete time column, reset index
df_nn


# In[ ]:


# Export it into .csv
f = "/content/gdrive/My Drive/Colab Notebooks/data/ARM_1hrlater.csv"
df_nn.to_csv(f)

