import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
matplotlib.use('Qt5Agg')  # TkAgg 백엔드 사용

# File paths
inputdir='./obs_data/peak/'
outputdir='./obs_data/peak/'
nc_file = inputdir+'v_new_data_before.nc'
output_nc_file_time = outputdir+'v_new_fft_time_before.nc'
output_nc_file_height =outputdir+ 'v_new_fft_height_before.nc'
output_nc_file_width = outputdir+'v_new_fft_width_before.nc'
output_nc_file_time2 = outputdir+'u_new_fft_time_before.nc'
output_nc_file_height2 = outputdir+'u_new_fft_height_before.nc'
output_nc_file_width2 = outputdir+'u_new_fft_width_before.nc'
boxstart=214
beforebox=boxstart-10
spongebuf=25
spinup=700
forplot=1700
# Function to perform Fourier transform only on the non-masked regions for both u_new and v_new
def perform_fourier_transform_with_mask(nc_file, output_nc_file_time, output_nc_file_height, output_nc_file_width, output_nc_file_time2,output_nc_file_height2,output_nc_file_width2):
    with nc.Dataset(nc_file, 'r') as ds:
        #v_new_data = ds.variables['v_new'][:24999,3:-3, 5:beforebox]
        #u_new_data = ds.variables['u_new'][:24999,3:-3, 5:beforebox]
        #flow_mask = ds.variables['flow_mask'][:24999,3:-3, 5:beforebox]
        v_new_data = ds.variables['v_new'][spinup:,3+spongebuf:-spongebuf-3, 5:beforebox]
        u_new_data = ds.variables['u_new'][spinup:,3+spongebuf:-spongebuf-3, 5:beforebox]
        flow_mask = ds.variables['flow_mask'][spinup:,3+spongebuf:-spongebuf-3, 5:beforebox]
        #v_new_data = ds.variables['v_new'][spinup:forplot,3+spongebuf:-spongebuf-3, 5:beforebox]
        #u_new_data = ds.variables['u_new'][spinup:forplot,3+spongebuf:-spongebuf-3, 5:beforebox]
        #flow_mask = ds.variables['flow_mask'][spinup:forplot,3+spongebuf:-spongebuf-3, 5:beforebox]
    print('1')
    # Initialize arrays for FFT results
    fft_shapes = (v_new_data.shape[0], v_new_data.shape[1], v_new_data.shape[2])
    v_new_fft_time_real = np.zeros(fft_shapes)
    v_new_fft_time_imag = np.zeros(fft_shapes)
    v_new_fft_height_real = np.zeros(fft_shapes)
    v_new_fft_height_imag = np.zeros(fft_shapes)
    v_new_fft_width_real = np.zeros(fft_shapes)
    v_new_fft_width_imag = np.zeros(fft_shapes)
    u_new_fft_time_real = np.zeros(fft_shapes)
    u_new_fft_time_imag = np.zeros(fft_shapes)
    u_new_fft_height_real = np.zeros(fft_shapes)
    u_new_fft_height_imag = np.zeros(fft_shapes)
    u_new_fft_width_real = np.zeros(fft_shapes)
    u_new_fft_width_imag = np.zeros(fft_shapes)

    print('2')
    # Perform FFT along the time dimension for each spatial location
    for i in range(v_new_data.shape[1]):
        for j in range(v_new_data.shape[2]):
            fft_result_v = np.fft.fft(v_new_data[:, i, j])
            fft_result_u = np.fft.fft(u_new_data[:, i, j])
            v_new_fft_time_real[:, i, j] = np.real(fft_result_v)
            v_new_fft_time_imag[:, i, j] = np.imag(fft_result_v)
            u_new_fft_time_real[:, i, j] = np.real(fft_result_u)
            u_new_fft_time_imag[:, i, j] = np.imag(fft_result_u)

    # Perform FFT along the height dimension for each time step
    for t in range(v_new_data.shape[0]):
        for j in range(v_new_data.shape[2]):
            fft_result_v = np.fft.fft(v_new_data[t, :, j])
            fft_result_u = np.fft.fft(u_new_data[t, :, j])
            v_new_fft_height_real[t, :, j] = np.real(fft_result_v)
            v_new_fft_height_imag[t, :, j] = np.imag(fft_result_v)
            u_new_fft_height_real[t, :, j] = np.real(fft_result_u)
            u_new_fft_height_imag[t, :, j] = np.imag(fft_result_u)

    # Perform FFT along the width dimension for each time step
    for t in range(v_new_data.shape[0]):
        for i in range(v_new_data.shape[1]):
            fft_result_v = np.fft.fft(v_new_data[t, i, :])
            fft_result_u = np.fft.fft(u_new_data[t, i, :])
            v_new_fft_width_real[t, i, :] = np.real(fft_result_v)
            v_new_fft_width_imag[t, i, :] = np.imag(fft_result_v)
            u_new_fft_width_real[t, i, :] = np.real(fft_result_u)
            u_new_fft_width_imag[t, i, :] = np.imag(fft_result_u)

    # Save the Fourier transformed data and flow mask to separate NetCDF files
    def save_fft_results(output_nc_file, fft_real, fft_imag, flow_mask):
        with nc.Dataset(output_nc_file, 'w', format='NETCDF4') as ds_out:
            ds_out.createDimension('time', fft_real.shape[0])
            ds_out.createDimension('height', fft_real.shape[1])
            ds_out.createDimension('width', fft_real.shape[2])
            ds_out.createVariable('fft_real', 'f4', ('time', 'height', 'width'))
            ds_out.createVariable('fft_imag', 'f4', ('time', 'height', 'width'))
            ds_out.createVariable('flow_mask', 'f4', ('time', 'height', 'width'))
            ds_out.variables['fft_real'][:] = fft_real
            ds_out.variables['fft_imag'][:] = fft_imag
            ds_out.variables['flow_mask'][:] = flow_mask
    print('3')
    save_fft_results(output_nc_file_time, v_new_fft_time_real, v_new_fft_time_imag, flow_mask)
    save_fft_results(output_nc_file_height, v_new_fft_height_real, v_new_fft_height_imag, flow_mask)
    save_fft_results(output_nc_file_width, v_new_fft_width_real, v_new_fft_width_imag, flow_mask)
    save_fft_results(output_nc_file_time2, u_new_fft_time_real, u_new_fft_time_imag, flow_mask)
    save_fft_results(output_nc_file_height2, u_new_fft_height_real, u_new_fft_height_imag, flow_mask)
    save_fft_results(output_nc_file_width2, u_new_fft_width_real, u_new_fft_width_imag, flow_mask)

# Function to compute the energy spectra from Fourier transformed data considering the flow mask
def compute_energy_spectra_with_mask(nc_file_time, nc_file_height, nc_file_width,nc_file_time2, nc_file_height2, nc_file_width2):
    def load_fft_results(nc_file):
        with nc.Dataset(nc_file, 'r') as ds:
            fft_real = ds.variables['fft_real'][:]
            fft_imag = ds.variables['fft_imag'][:]
            flow_mask = ds.variables['flow_mask'][:]
        return fft_real, fft_imag, flow_mask

    v_new_fft_time_real, v_new_fft_time_imag, flow_mask = load_fft_results(nc_file_time)
    v_new_fft_height_real, v_new_fft_height_imag, _ = load_fft_results(nc_file_height)
    v_new_fft_width_real, v_new_fft_width_imag, _ = load_fft_results(nc_file_width)
    u_new_fft_time_real, u_new_fft_time_imag, flow_mask = load_fft_results(nc_file_time2)
    u_new_fft_height_real, u_new_fft_height_imag, _ = load_fft_results(nc_file_height2)
    u_new_fft_width_real, u_new_fft_width_imag, _ = load_fft_results(nc_file_width2)

    # Compute magnitude of the complex Fourier coefficients
    def compute_magnitude(fft_real, fft_imag):
        return np.sqrt(fft_real**2 + fft_imag**2)

    v_new_fft_time_magnitude = compute_magnitude(v_new_fft_time_real, v_new_fft_time_imag)
    v_new_fft_height_magnitude = compute_magnitude(v_new_fft_height_real, v_new_fft_height_imag)
    v_new_fft_width_magnitude = compute_magnitude(v_new_fft_width_real, v_new_fft_width_imag)
    u_new_fft_time_magnitude = compute_magnitude(u_new_fft_time_real, u_new_fft_time_imag)
    u_new_fft_height_magnitude = compute_magnitude(u_new_fft_height_real, u_new_fft_height_imag)
    u_new_fft_width_magnitude = compute_magnitude(u_new_fft_width_real, u_new_fft_width_imag)

    # Compute the energy spectrum for spatial dimensions (wave number)
    def compute_energy_spectrum(fft_magnitude, flow_mask, axis):
        energy_spectrum = np.sum(fft_magnitude**2 * flow_mask/2, axis=axis)
        energy_spectrum = energy_spectrum / np.sum(flow_mask, axis=axis)  # normalize by the number of non-masked elements
        return np.clip(energy_spectrum, 1e-10, None)  # Ensure positive values for log scale

    energy_wave_number_height_v = compute_energy_spectrum(v_new_fft_height_magnitude, flow_mask, (0, 2))
    energy_wave_number_width_v = compute_energy_spectrum(v_new_fft_width_magnitude, flow_mask, (0, 1))
    energy_frequency_v = compute_energy_spectrum(v_new_fft_time_magnitude, flow_mask, (1, 2))
    energy_wave_number_height_u = compute_energy_spectrum(u_new_fft_height_magnitude, flow_mask, (0, 2))
    energy_wave_number_width_u = compute_energy_spectrum(u_new_fft_width_magnitude, flow_mask, (0, 1))
    energy_frequency_u = compute_energy_spectrum(u_new_fft_time_magnitude, flow_mask, (1, 2))

    return energy_wave_number_height_v, energy_wave_number_width_v, energy_frequency_v, energy_wave_number_height_u, energy_wave_number_width_u, energy_frequency_u

def find_all_peaks(frequencies, energies, height_threshold=1, distance=10, prominence=1):
    """
    주어진 주파수 배열과 에너지 배열에서 모든 피크를 찾고 해당 주파수와 에너지를 반환합니다.

    Parameters:
    - frequencies: 주파수 배열
    - energies: 에너지 배열
    - height_threshold: 피크로 인식할 최소 에너지 값
    - distance: 피크 간 최소 거리 (인덱스 기준)
    - prominence: 피크의 돌출 정도 최소값

    Returns:
    - peak_frequencies: 피크가 발생한 주파수들
    - peak_energies: 피크 에너지 값들
    """
    # find_peaks를 이용해 에너지 배열에서 피크 탐색
    energies = np.log(energies)
    peak_indices, _ = find_peaks(energies, height=height_threshold, distance=distance, prominence=prominence)

    # 피크 인덱스에 해당하는 주파수와 에너지 값 반환
    peak_frequencies = frequencies[peak_indices]
    peak_energies = energies[peak_indices]
    peak_energies = np.exp(peak_energies)

    return peak_frequencies, peak_energies

# Function to plot energy spectra for both u_new and v_new
def plot_energy_spectra(energy_wave_number_height_v, energy_wave_number_width_v, energy_frequency_v,
                        energy_wave_number_height_u, energy_wave_number_width_u, energy_frequency_u):
    plt.figure(figsize=(21, 12))

    # Plot energy vs. frequency (from time FFT) for v_new
    plt.subplot(2, 3, 1)
    frequencies = np.fft.fftfreq(energy_frequency_v.shape[0])
    frequencies = frequencies[frequencies > 0]  # consider only positive frequencies
    energy_frequency_v = energy_frequency_v[1:frequencies.size + 1]
    distance_threshold = len(frequencies) // 80
    peak_frequencies_v, peak_energies_v = find_all_peaks(frequencies, energy_frequency_v, height_threshold=-2, distance=distance_threshold, prominence=1)
    plt.plot(frequencies, energy_frequency_v)
    plt.yscale('log')
    plt.title('Energy vs. Frequency (Time FFT) for v_new')
    plt.xlabel('Frequency')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")
    plt.scatter(peak_frequencies_v, peak_energies_v, color='red')
    print('Peak frequencies and energies for v_new:')
    for peak_freq_v, peak_eng_v in zip(peak_frequencies_v, peak_energies_v):
        print(f'Peak frequency: {peak_freq_v}, Peak energy: {peak_eng_v}')
    print('=' * 50)

    # Plot energy vs. wave number (from height FFT) for v_new
    plt.subplot(2, 3, 2)
    wave_numbers_height = np.fft.fftfreq(energy_wave_number_height_v.shape[0])
    wave_numbers_height = wave_numbers_height[wave_numbers_height > 0]  # consider only positive frequencies
    energy_wave_number_height_v = energy_wave_number_height_v[1:wave_numbers_height.size + 1]
    plt.plot(wave_numbers_height, energy_wave_number_height_v)
    plt.yscale('log')
    plt.title('Energy vs. Wave Number (Height FFT) for v_new')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")

    # Plot energy vs. wave number (from width FFT) for v_new
    plt.subplot(2, 3, 3)
    wave_numbers_width = np.fft.fftfreq(energy_wave_number_width_v.shape[0])
    wave_numbers_width = wave_numbers_width[wave_numbers_width > 0]  # consider only positive frequencies
    energy_wave_number_width_v = energy_wave_number_width_v[1:wave_numbers_width.size + 1]
    plt.plot(wave_numbers_width, energy_wave_number_width_v)
    plt.yscale('log')
    plt.title('Energy vs. Wave Number (Width FFT) for v_new')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")

    # Plot energy vs. frequency (from time FFT) for u_new
    plt.subplot(2, 3, 4)
    frequencies = np.fft.fftfreq(energy_frequency_u.shape[0])
    frequencies = frequencies[frequencies > 0]  # consider only positive frequencies
    energy_frequency_u = energy_frequency_u[1:frequencies.size + 1]
    peak_frequencies_u, peak_energies_u = find_all_peaks(frequencies, energy_frequency_u, height_threshold=-2, distance=distance_threshold, prominence=1)
    plt.plot(frequencies, energy_frequency_u)
    plt.yscale('log')
    plt.title('Energy vs. Frequency (Time FFT) for u_new')
    plt.xlabel('Frequency')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")
    plt.scatter(peak_frequencies_u, peak_energies_u, color='red')
    print('Peak frequencies and energies for u_new:')
    for peak_freq_u, peak_eng_u in zip(peak_frequencies_u, peak_energies_u):
        print(f'Peak frequency: {peak_freq_u}, Peak energy: {peak_eng_u}')
    print('=' * 50)

    # Plot energy vs. wave number (from height FFT) for u_new
    plt.subplot(2, 3, 5)
    wave_numbers_height = np.fft.fftfreq(energy_wave_number_height_u.shape[0])
    wave_numbers_height = wave_numbers_height[wave_numbers_height > 0]  # consider only positive frequencies
    energy_wave_number_height_u = energy_wave_number_height_u[1:wave_numbers_height.size + 1]
    plt.plot(wave_numbers_height, energy_wave_number_height_u)
    plt.yscale('log')
    plt.title('Energy vs. Wave Number (Height FFT) for u_new')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")

    # Plot energy vs. wave number (from width FFT) for u_new
    plt.subplot(2, 3, 6)
    wave_numbers_width = np.fft.fftfreq(energy_wave_number_width_u.shape[0])
    wave_numbers_width = wave_numbers_width[wave_numbers_width > 0]  # consider only positive frequencies
    energy_wave_number_width_u = energy_wave_number_width_u[1:wave_numbers_width.size + 1]
    plt.plot(wave_numbers_width, energy_wave_number_width_u)
    plt.yscale('log')
    plt.title('Energy vs. Wave Number (Width FFT) for u_new')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('./obs_data/peak/energy_spectra_before.png')


# Perform Fourier transform considering the flow mask and save the results
perform_fourier_transform_with_mask(nc_file, output_nc_file_time, output_nc_file_height, output_nc_file_width,output_nc_file_time2, output_nc_file_height2, output_nc_file_width2)
print('4')
# Compute energy spectra considering the flow mask
(energy_wave_number_height_v, energy_wave_number_width_v, energy_frequency_v,
 energy_wave_number_height_u, energy_wave_number_width_u, energy_frequency_u) = compute_energy_spectra_with_mask(output_nc_file_time, output_nc_file_height, output_nc_file_width,output_nc_file_time2, output_nc_file_height2, output_nc_file_width2)
print('5')
# Plot energy spectra
plot_energy_spectra(energy_wave_number_height_v, energy_wave_number_width_v, energy_frequency_v,
                    energy_wave_number_height_u, energy_wave_number_width_u, energy_frequency_u)

