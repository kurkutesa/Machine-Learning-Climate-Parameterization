# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, ceil
import seaborn as sns

import xarray as xr
#import tensorflow as tf
print('All packages imported.')

# Read data

DIR = '../data/forNN/'
f = DIR + 'ARM_no_dropna.cdf'
DS = xr.open_dataset(f)
DS = DS.drop(['z10','z2'])

# Build multiple DataArray to collapse pressure coord

precip = DS['prec_sfc'] # the only DataArray with no pressure coord
da_list = ['precip'] # initialize list of DataArray

plev = DS['p'].values.astype(np.int32) # array of pressure level
var_p_list = ['T_p','rh_p'] # array of variables with pressure coord

for _var in var_p_list:
  for _p in plev:
    new_var_str = (_var + str(_p)) # new variable name (in string)
    da_list.append(new_var_str) # append the new variable
    exec("{} = DS[_var].sel(p=_p).drop('p')".format(new_var_str)) # assign DataArray
    exec("{} = {}.rename('{}')".format(new_var_str,new_var_str,new_var_str)) # rename data variables to avoid merge collision

# Merge DataArray and Build a new DataSet
DS_new = xr.Dataset()
for _var in da_list:
  exec("DS_new = xr.merge([DS_new,{}])".format(_var))

# Convert DataSet to dataframe
df = DS_new.to_dataframe().reset_index() # convert index (time) to a column

#df['month'] = list(map(lambda x:x.month, df['time']))
df['hour'] = list(map(lambda x:x.hour, df['time']))
print('Ready!')
sns_plot = sns.pairplot(df.dropna().drop(columns=['time']),
                        vars = ['prec_sfc', 'T_p1000', 'rh_p1000'],
                        hue = 'hour',
                        palette = 'bright',
                        diag_kind = 'kde',
                        plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                        size = 4)
sns_plot.savefig('pairplot_hour.png')
