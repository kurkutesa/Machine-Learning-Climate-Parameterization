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

def sum_in_year_week(DA):
    # Index of Year-week starts at Jan 1991
    week_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 52 + DA.time.dt.weekofyear.to_index()
    # Assign newly defined Year-week to coordinates, then group by it, then take the SUM
    return DA.assign_coords(year_week = week_cnt_1991).groupby('year_week').sum()

def mean_in_year_week(DA):
    week_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 52 + DA.time.dt.weekofyear.to_index()
    return DA.assign_coords(year_week = week_cnt_1991).groupby('year_week').mean()

# Import Vertical Velocity from 500hPa data from ERA Interim
file = 'data/vv_500hpa.nc'
DS = xr.open_dataset(file).chunk() # Convert it to dask array
vv = DS.w.dropna(dim='time')
vv_weekly = mean_in_year_week(vv)
toc()

# Import Precipitation data from ARM
mf = '../ARM-PNG/data/twparmbeatmC1.c1.*.000000.cdf'
DS2 = xr.open_mfdataset(mf)
precip = DS2.prec_sfc.dropna(dim='time')
precip_weekly = sum_in_year_week(precip)
toc()

# Merge them to drop missing data
DS_weekly = xr.merge([vv_weekly,precip_weekly]).dropna(dim='year_week')
#x = DS_weekly.prec_sfc.values
#y = DS_weekly.w.values

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
plt.xlabel('Lag (week)')
plt.ylabel('Normalized Cross-correlation')
plt.title('Correlation between Precipitation and Vertical Velocity')
plt.savefig('fig/cc.weekly.eps')
'''
DS_weekly_in = DS_weekly.where(DS_weekly.w < 0.15, drop=True).where(DS_weekly.w > -0.15, drop=True)
x = DS_weekly_in.prec_sfc.values
y = DS_weekly_in.w.values
toc()

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
plt.xlabel('Weekly Precipitation (mm)')
plt.ylabel('Weekly Average Vertical Velocity (Pa/s)')
plt.title('Weekly Average in Manus, PNG')
plt.savefig('fig/no_outliers_scheme/vv_500hpa.precip.WEEKLY.regr.eps')
toc()
