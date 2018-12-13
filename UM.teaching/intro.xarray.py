# XARRAY Basics - Eden Au
import numpy as np # pkg for high level maths
import xarray as xr # for handling NetCDF files

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # for plotting

# DataSet and DataArray are two data types in xarray

link = '../nc/apa/xoffaa.pal4dec.nc'
DS = xr.open_dataset(link)


# Use ncdump or ipython to check variables in the DataSet first

# Extract a DataArray, say the air_temperature
# (don't do snowfall_amount, lots of zeros)
airtemp = DS.air_temperature

# You will find that it is a DataArray with one variable, and
# three coordinates: time, latitude, longitude
# You can check the coordinates by
# airtemp.coords['longitude']

# Use .sel() to select the ones that you need (e.g. latlon of Edinburgh)
# Try to guess why we need method='nearest'
# Why longitude is negative? Go check the definition of it
airtemp_edin = airtemp.sel(latitude = 55.9533, longitude = -3.1883, method = 'nearest')
# This DataArray now has one variable with one coordinate: time

# 1 var, 1 coord -> 2D line plot
airtemp_edin.plot()

# Always save/show your plot!
#plt.show()
plt.savefig('airtemp_edinburgh.jpg')
