import time
# MATLAB like tic toc
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
def tic():
    toc(False)
tic()

#import numpy as np
#from sklearn import linear_model
#import matplotlib.pyplot as plt
import xarray as xr
toc()

# Import data
#mf = '../data/forNN/erainterim.*.201314.nc'
#DS = xr.open_mfdataset(mf)

mf = '../data/ARM/PNG/twparmbeatmC1.c1.*.000000.cdf'
DS = xr.open_mfdataset(mf)
T_p = DS.T_p.dropna(dim='time')
rh_p = DS.rh_p.dropna(dim='time')
u_p = DS.u_p.dropna(dim='time')
v_p = DS.v_p.dropna(dim='time')
prec_sfc = DS.prec_sfc.dropna(dim='time')
toc()

# Import Vertical Velocity from 500hPa data from ERA Interim
file = '../data/ERAInterim/vv500hpa.nc'
DS2 = xr.open_dataset(file).chunk() # Convert it to dask array
w = DS2.w.dropna(dim='time')
toc()

# Merge them to drop missing data
DS_new = xr.merge([T_p,rh_p,u_p,v_p,prec_sfc,w]).dropna(dim='time')
new_path = '../data/forNN/NN_input.cdf'
DS_new.to_netcdf(new_path)
toc()
