import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


#File path
inputdir='/home/binpyo/data/before/240722/'
outputdir='/home/binpyo/data/before/240722/'



np.random.seed(42)  # Set a random seed for reproducibilit

# Function to read adjustment energy from the NetCDF file
def read_adjustment_energy(nc_file):
    with nc.Dataset(nc_file, 'r') as ds:
        frequencies = ds.variables['frequency'][:]
        adjustment_energy = ds.variables['adjustment_energy'][:]
    return frequencies, adjustment_energy

# Function to perform inverse FFT
def inverse_fft(real_part, imag_part):
    complex_array = real_part + 1j * imag_part
    return np.fft.ifft(complex_array).real

# Read adjustment energy from the NetCDF file
adjustment_energy_nc_file = inputdir+'u_adjustment.nc'
frequencies, energy_frequency_u = read_adjustment_energy(adjustment_energy_nc_file)
adjustment_energy_nc_file = inputdir+'v_adjustment.nc'
frequencies, energy_frequency_v = read_adjustment_energy(adjustment_energy_nc_file)

#min
energy_frequency_u = np.clip(energy_frequency_u, 0, None)
energy_frequency_v = np.clip(energy_frequency_v, 0, None)

# Generate random values for the real and imaginary parts of u and v
rand_u_real = np.random.rand(len(energy_frequency_u))
rand_v_real = np.random.rand(len(energy_frequency_v))

u_real = np.sqrt(rand_u_real * 2 * energy_frequency_u)
u_imag = np.sqrt((1 - rand_u_real) * 2 * energy_frequency_u)
v_real = np.sqrt(rand_v_real * 2 * energy_frequency_v)
v_imag = np.sqrt((1 - rand_v_real) * 2 *  energy_frequency_v)


# Apply a Hann w22indow to the frequency domain data
#hann_window = np.hanning(len(u_real))
#u_real *= hann_window
#u_imag *= hann_window
#v_real *= hann_window
#v_imag *= hann_window
print(np.average(u_real**2+v_real**2))
#print(np.average(u_real))
#print(np.average(np.abs(u_real))
n=len(energy_frequency_u)

print(energy_frequency_u[0])



#u_adj=np.zeros(n)
#v_adj=np.zeros(n)
#for i in range(n):
#    sum_real_u = 0
#    sum_real_v = 0
#    for j in range(n):
#        angle = 2 * np.pi * j * i / n
#        sum_real_u += u_real[j] * np.cos(angle) + u_imag[j] * np.sin(angle)
#        sum_real_v += v_real[j] * np.cos(angle) + v_imag[j] * np.sin(angle)
#    u_adj[i] = sum_real_u
#    v_adj[i] = sum_real_v

#u_adj=u_adj/n
#v_adj=v_adj/n


#u_complex = u_real + 1j * u_imag
#u_complex_fft = np.zeros(n+1,dtype=complex)
#u_complex_fft[1:n+1] = u_complex
#u_complex_fft[n+1:] = u_complex[::-1]
#u_adj=np.fft.ifft(u_complex_fft).real

#v_complex = v_real + 1j * v_imag
#v_complex_fft = np.zeros(n+1,dtype=complex)
#v_complex_fft[1:n+1] = v_complex
#v_complex_fft[n+1:] = v_complex[::-1]
#v_adj=np.fft.ifft(v_complex_fft).real

# Perform inverse FFT to get u_adj and v_adj
u_adj = inverse_fft(u_real, u_imag)
v_adj = inverse_fft(v_real, v_imag)


time_dim = len(u_adj)
u_obs_adj = np.zeros((time_dim, 1, 1))
v_obs_adj = np.zeros((time_dim, 1, 1))

# Assign u_adj and v_adj to the appropriate channels
u_obs_adj[:, 0, 0] = u_adj
v_obs_adj[:, 0, 0] = v_adj


# Save the v_obs_adj array to a new NetCDF file
output_nc_file = outputdir+'v_obs_adj.nc'
with nc.Dataset(output_nc_file, 'w', format='NETCDF4') as ds:
    ds.createDimension('time', time_dim)
    ds.createDimension('height', 1)
    ds.createDimension('width', 1)

    time_var = ds.createVariable('time', 'f4', ('time',))
    height_var = ds.createVariable('height', 'f4', ('height',))
    width_var = ds.createVariable('width', 'f4', ('width',))
    u_obs_adj_var = ds.createVariable('u_obs_adj', 'f4', ('time', 'height', 'width'))
    v_obs_adj_var = ds.createVariable('v_obs_adj', 'f4', ('time', 'height', 'width'))

    time_var[:] = np.arange(time_dim)
    height_var[:] = np.arange(1)
    width_var[:] = np.arange(1)
    u_obs_adj_var[:] = u_obs_adj
    v_obs_adj_var[:] = v_obs_adj


# Plotting the results
plt.figure(figsize=(14, 7))

# Plot u_adj
plt.subplot(2, 1, 1)
plt.plot(u_adj, label='u_adj')
plt.xlabel('Time')
plt.ylabel('u_adj')
plt.ylim([-0.005,0.005])
plt.title('u_adj Time Series')
plt.legend()
plt.grid(True)

# Plot v_adj
plt.subplot(2, 1, 2)
plt.plot(v_adj, label='v_adj')
plt.xlabel('Time')
plt.ylabel('v_adj')
plt.ylim([-0.005,0.005])
plt.title('v_adj Time Series')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

