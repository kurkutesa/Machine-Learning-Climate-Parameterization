import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cartopy.crs as ccrs # CARTOgraphy for PYthon

DS = xr.open_dataset('../nc/apm/xoffaa.pml4dec.nc')

# We plotted temporal data before, we now look into spatial data
sap = DS.surface_air_pressure

# Draw coastlines of the Earth
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

sap.plot()
plt.savefig('sap.jpg')
