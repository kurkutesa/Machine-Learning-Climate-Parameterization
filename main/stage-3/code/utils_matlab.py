import matlab.engine
import xarray as xr
#% Usage:
#%         function [wout] = convert_h2o(p,t,winp,unitinp,unitout)
#% Input:  p       - Pressure in hPa
#%         t       - Temperature in K
#%         winp    - Water Vapor concentration in
#%         unitinp - Input units
#%         unitout - Output units
def add_column_water(DS):
    p = matlab.double(list(DS['p'].values))
    T_p_all, rh_p_all = [list(DS[var].values) for var in ['T_p', 'rh_p']]
    engine = matlab.engine.start_matlab()

    col_water = []
    for T_p, rh_p in zip(T_p_all, rh_p_all):
        col_water.append(engine.convert_h2o(p,
                                            matlab.double(list(T_p)),
                                            matlab.double(list(rh_p)),
                                            'H', 'C'))
    col_water_da = xr.DataArray(col_water,
                                   coords={'time':DS.time},
                                   dims='time',
                                   name='col_water')

    return xr.merge([DS, col_water_da])
