import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
print('Import completed')

mf = 'data/era_interim.vv_500hpa.*.nc'
DS = xr.open_mfdataset(mf)
print('DataSet load completed')

# Load it otherwise memory error in write stage
vv_PNG = DS.w.sel(latitude = -2.1, longitude = 147.4, method = 'nearest').dropna(dim='time').load()
print('DataArray prep completed')

write_file = 'data/vv_500hpa.nc'
vv_PNG.to_netcdf(write_file)
print('Export completed')

#vv_PNG.plot()
#plt.savefig('vv_PNG.jpg')
