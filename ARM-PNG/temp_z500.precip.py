import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Contract the DataArray by taking mean for each Year-Month
def mean_in_year_month(DA):
    # Index of Year-Month starts at Jan 1991
    month_cnt_1991 = (DA.time.dt.year.to_index() - 1991) * 12 + DA.time.dt.month.to_index()
    # Assign newly defined Year-Month to coordinates, then group by it, then take the mean
    return DA.assign_coords(year_month = month_cnt_1991).groupby('year_month').mean()

def mean_in_year(DA):
    return DA.assign_coords(year = DA.time.dt.year.to_index()).groupby('year').mean()

# Load DataSet
mf = 'data/twparmbeatmC1.c1.*.000000.cdf'
DS = xr.open_mfdataset(mf)
# xr.open_dataset would suffice if there is only one NetCDF file

# Use ncdump or ipython to check variables in DS first

# Extract Dry Bulb Temperature in z-coordinate (T_z)
# Select the altitude nearest to 500m above surface
# Drop NaN, convert to Celcius
temp_z500 = DS.T_z.sel(z=500,method='nearest').dropna(dim='time') - 273.15  # or .ffill(dim='time')
# Extract Precipitation Rate
precip = DS.prec_sfc.dropna(dim='time')


temp_z500_monthly = mean_in_year_month(temp_z500)
precip_monthly = mean_in_year_month(precip)
# Merge them to a single DataSet, drop NaN
DS_monthly = xr.merge([temp_z500_monthly,precip_monthly]).dropna(dim='year_month')

# Yearly
'''
temp_z500_yearly = mean_in_year(temp_z500)
precip_yearly = mean_in_year(precip)
# Merge them to a single DataSet, drop NaN
DS_yearly = xr.merge([temp_z500_yearly,precip_yearly]).dropna(dim='year')
'''


plt.scatter(DS_monthly.T_z , DS_monthly.prec_sfc)
plt.xlabel('Temperature 500m above surface (C)')
plt.ylabel('Precipitation (mm/hour)')
plt.title('Monthly Average in Manus, PNG')
plt.show()
