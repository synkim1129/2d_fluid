import numpy as np
import netCDF4 as nc

# File paths
inputdir1='/home/binpyo/data/before/240722/'
inputdir2='/home/binpyo/data/before/240722/'
outputdir='/home/binpyo/data/before/240722/'

v_new_data_file = inputdir1+'v_new_data.nc'
v_obs_adj_file = inputdir2+'v_obs_adj.nc'
output_nc_file = outputdir+'v_obs.nc'

# Read data from v_new_data.nc
with nc.Dataset(v_new_data_file, 'r') as ds_v_new:
    u_new = ds_v_new.variables['u_new'][:]
    v_new = ds_v_new.variables['v_new'][:]
    flow_mask = ds_v_new.variables['flow_mask'][:]

# Read adjustment data from v_obs_adj.nc
with nc.Dataset(v_obs_adj_file, 'r') as ds_obs_adj:
    time = ds_obs_adj.variables['time'][:]
    u_obs_adj = ds_obs_adj.variables['u_obs_adj'][:]
    v_obs_adj = ds_obs_adj.variables['v_obs_adj'][:]

# Adjust the data
timelen=len(time)
obs_shapes = (timelen, u_new.shape[1], u_new.shape[2])
u_obs= np.zeros(obs_shapes)
v_obs= np.zeros(obs_shapes)
for i in range(u_new.shape[1]):
    for j in range(u_new.shape[2]):
        u_obs[:,i,j]=u_new[:timelen,i,j]+u_obs_adj[:,0,0]
        v_obs[:,i,j]=v_new[:timelen,i,j]+v_obs_adj[:,0,0]
    

# Save the combined data to a new NetCDF file
with nc.Dataset(output_nc_file, 'w', format='NETCDF4') as ds_out:
    # Create dimensions
    ds_out.createDimension('time', timelen)
    ds_out.createDimension('height', u_new.shape[1])
    ds_out.createDimension('width', u_new.shape[2])

    # Create variables
    time_var = ds_out.createVariable('time', 'i4', ('time',))
    u_obs_var = ds_out.createVariable('u_obs', 'f4', ('time', 'height', 'width'))
    v_obs_var = ds_out.createVariable('v_obs', 'f4', ('time', 'height', 'width'))

    # Assign data to variables
    time_var[:] = time
    u_obs_var[:] = u_obs
    v_obs_var[:] = v_obs

print(f'Successfully saved adjusted data to {output_nc_file}')

