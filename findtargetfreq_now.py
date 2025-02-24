import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용

# File paths
inputdir='/home/teamai/data/diff1000ep240/just/'
nc_file = inputdir+'v_new_data.nc'
boxstart=214
#boxstart=210
#beforebox=boxstart-20
beforebox=boxstart
spongebuf=25
spinup=1000
filterfreq=0.002
# Function to perform Fourier transform only on the non-masked regions for both u_new and v_new
with nc.Dataset(nc_file, 'r') as ds:
    v_new_data = ds.variables['v_new'][spinup:,3+spongebuf:-spongebuf-3, 5:beforebox]
    u_new_data = ds.variables['u_new'][spinup:,3+spongebuf:-spongebuf-3, 5:beforebox]
print('1')
fft_shapes = (v_new_data.shape[0], v_new_data.shape[1], v_new_data.shape[2])
v_new_fft_time_real = np.zeros(fft_shapes)
v_new_fft_time_imag = np.zeros(fft_shapes)
u_new_fft_time_real = np.zeros(fft_shapes)
u_new_fft_time_imag = np.zeros(fft_shapes)
for i in range(v_new_data.shape[1]):
    for j in range(v_new_data.shape[2]):
        fft_result_v = np.fft.fft(v_new_data[:, i, j])
        fft_result_u = np.fft.fft(u_new_data[:, i, j])
        v_new_fft_time_real[:, i, j] = np.real(fft_result_v)
        v_new_fft_time_imag[:, i, j] = np.imag(fft_result_v)
        u_new_fft_time_real[:, i, j] = np.real(fft_result_u)
        u_new_fft_time_imag[:, i, j] = np.imag(fft_result_u)

def compute_magnitude(fft_real, fft_imag):
    return np.sqrt(fft_real**2 + fft_imag**2)

v_new_fft_time_magnitude = compute_magnitude(v_new_fft_time_real, v_new_fft_time_imag)
u_new_fft_time_magnitude = compute_magnitude(u_new_fft_time_real, u_new_fft_time_imag)
def compute_energy_spectrum(fft_magnitude, axis):
    energy_spectrum = np.sum(fft_magnitude**2/2, axis=axis)
    return np.clip(energy_spectrum, 1e-10, None)  # Ensure positive values for log scale

energy_frequency_v = compute_energy_spectrum(v_new_fft_time_magnitude, (1, 2))
energy_frequency_u = compute_energy_spectrum(u_new_fft_time_magnitude, (1, 2))

frequencies = np.fft.fftfreq(energy_frequency_v.shape[0],d=4)
frequencies = frequencies[frequencies > 0]  # consider only positive frequencies

energy_frequency_v = energy_frequency_v[1:frequencies.size + 1]
energy_frequency_u = energy_frequency_u[1:frequencies.size + 1]
valid_indices = np.where(frequencies > filterfreq)[0]  # filterfreq보다 큰 주파수의 인덱스 찾기
filtered_frequencies = frequencies[valid_indices]  # filterfreq보다 큰 주파수만 남김
filtered_energy_frequency_v = energy_frequency_v[valid_indices]  # 해당 에너지 스펙트럼만 남김
filtered_energy_frequency_u = energy_frequency_u[valid_indices]  # 해당 에너지 스펙트럼만 남김


print(filtered_frequencies[np.argmax(filtered_energy_frequency_u[:])])
print(filtered_frequencies[np.argmax(filtered_energy_frequency_v[:])])
 
targetfreq=filtered_frequencies[np.argmax(filtered_energy_frequency_v[:])]
print(f"TARGET_FREQ={targetfreq:.5f}")
