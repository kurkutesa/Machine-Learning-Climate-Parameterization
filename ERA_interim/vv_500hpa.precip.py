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

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
toc()

def sum_in_year_month(DA):
    # Index of Year-Day starts at Jan 1991
    month_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 12 + DA.time.dt.month.to_index()
    # Assign newly defined Year-Day to coordinates, then group by it, then take the SUM
    return DA.assign_coords(year_month = month_cnt_1991).groupby('year_month').sum()

def mean_in_year_month(DA):
    month_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 12 + DA.time.dt.month.to_index()
    return DA.assign_coords(year_month = month_cnt_1991).groupby('year_month').mean()

# Import Vertical Velocity from 500hPa data from ERA Interim
file = 'data/vv_500hpa.nc'
DS = xr.open_dataset(file).chunk() # Convert it to dask array
vv = DS.w.dropna(dim='time')
vv_monthly = mean_in_year_month(vv)
toc()

# Import Precipitation data from ARM
mf = '../ARM-PNG/data/twparmbeatmC1.c1.*.000000.cdf'
DS2 = xr.open_mfdataset(mf)
precip = DS2.prec_sfc.dropna(dim='time')
precip_monthly = sum_in_year_month(precip)
toc()

# Merge them to drop missing data
DS_monthly = xr.merge([vv_monthly,precip_monthly]).dropna(dim='year_month')
x = DS_monthly.prec_sfc.values
y = DS_monthly.w.values
x_norm = x-x.mean()
y_norm = y-y.mean()
cc = np.correlate(x_norm,y_norm,'full')
lag = range(-len(x)+1,len(x))
toc()

plt.figure()
plt.plot(lag,cc)
plt.xlabel('Lag (month)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Precipitation and Vertical Velocity')
plt.savefig('fig/cc.eps')

plt.figure()
plt.scatter(x,y)
plt.xlabel('Monthly Precipitation (mm)')
plt.ylabel('Monthly Average Vertical Velocity (Pa/s)')
plt.title('Monthly Average in Manus, PNG')
plt.savefig('fig/vv_500hpa.precip.eps')
toc()
