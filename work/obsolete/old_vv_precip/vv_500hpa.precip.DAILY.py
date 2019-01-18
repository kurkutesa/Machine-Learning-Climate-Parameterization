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
from sklearn import linear_model
import matplotlib.pyplot as plt
import xarray as xr
toc()

def sum_in_year_day(DA):
    # Index of Year-Day starts at Jan 1991
    day_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 366 + DA.time.dt.dayofyear.to_index()
    # Assign newly defined Year-Day to coordinates, then group by it, then take the SUM
    return DA.assign_coords(year_day = day_cnt_1991).groupby('year_day').sum()

def mean_in_year_day(DA):
    day_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 366 + DA.time.dt.dayofyear.to_index()
    return DA.assign_coords(year_day = day_cnt_1991).groupby('year_day').mean()

# Import Vertical Velocity from 500hPa data from ERA Interim
file = 'data/vv_500hpa.nc'
DS = xr.open_dataset(file).chunk() # Convert it to dask array
vv = DS.w.dropna(dim='time')
vv_daily = mean_in_year_day(vv)
toc()

# Import Precipitation data from ARM
mf = '../ARM-PNG/data/twparmbeatmC1.c1.*.000000.cdf'
DS2 = xr.open_mfdataset(mf)
precip = DS2.prec_sfc.dropna(dim='time')
precip_daily = sum_in_year_day(precip)
toc()

# Merge them to drop missing data
DS_daily = xr.merge([vv_daily,precip_daily]).dropna(dim='year_day')
#x = DS_daily.prec_sfc.values
#y = DS_daily.w.values

# Find Cross-correlation between time-series variables
'''
x_norm = x-x.mean()
y_norm = y-y.mean()
cc = np.correlate(x_norm,y_norm,'full')
cc = cc / max(abs(cc))
lag = range(-len(x)+1,len(x))
toc()

plt.figure()
plt.plot(lag,cc)
plt.xlabel('Lag (day)')
plt.ylabel('Normalized Cross-correlation')
plt.title('Correlation between Precipitation and Vertical Velocity')
plt.savefig('fig/cc.DAILY.eps')
'''
DS_daily_in = DS_daily.where(DS_daily.prec_sfc >20, drop=True).where(DS_daily.w < 0.5, drop=True).where(DS_daily.w > -0.5, drop=True)
x = DS_daily_in.prec_sfc.values
y = DS_daily_in.w.values

# Try RANSAC
ransac = linear_model.RANSACRegressor()
ransac.fit(x.reshape(-1,1), y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(x.min(), x.max())[:, np.newaxis] # vertical array
line_y = ransac.predict(line_x)

plt.figure()
plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_x, line_y, color='cornflowerblue', linewidth=2, label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel('Daily Precipitation (mm)')
plt.ylabel('Daily Average Vertical Velocity (Pa/s)')
plt.title('Daily Average in Manus, PNG')
plt.savefig('fig/no_outliers_scheme/vv_500hpa.precip.regr.eps')
toc()
