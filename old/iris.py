from netCDF4 import Dataset
import iris

data = Dataset('twparmbeatmC1.c1.19960101.000000.cdf')
print(data.file_format)

#print(dataset.dimensions.keys())#['time','range','p','z']
#print(dataset.dimensions['z'])#size 8784*2*37*512
#print(dataset.variables.keys())
#(['base_time', 'time_offset', 'time', 'time_bounds', 'time_frac', 'p', 'p_bounds',
#'z', 'z_bounds', 'z10', 'z2', 'u_sfc', 'v_sfc', 'T_sfc', 'rh_sfc', 'p_sfc', 'prec_sfc',
#'T_p', 'T_z', 'Td_p', 'Td_z', 'rh_p', 'rh_z', 'u_p', 'u_z', 'v_p', 'v_z', 'lat', 'lon', 'alt'])

ncdump.data


data.close()
