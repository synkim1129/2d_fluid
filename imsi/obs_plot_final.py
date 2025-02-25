import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

inputdir='/home/binpyo/data/before/240722/'
outputdir='/home/binpyo/data/before/240722/'

# File paths
input_nc_file1 = inputdir+'v_new_fft_time.nc'
input_nc_file2 = inputdir+'v_new_fft_height.nc'
input_nc_file3 = inputdir+'v_new_fft_width.nc'
filename= outputdir+'v_energy_time.nc'
# Function to compute the energy spectra from Fourier transformed data considering the flow mask
def compute_energy_spectra_with_mask(nc_file1, nc_file2, nc_file3):
    with nc.Dataset(nc_file1, 'r') as ds:
        v_new_fft_time_real = ds.variables['fft_real'][:]
        v_new_fft_time_imag = ds.variables['fft_imag'][:]
    with nc.Dataset(nc_file2, 'r') as ds:
        v_new_fft_height_real = ds.variables['fft_real'][:]
        v_new_fft_height_imag = ds.variables['fft_imag'][:]
    with nc.Dataset(nc_file3, 'r') as ds:
        v_new_fft_width_real = ds.variables['fft_real'][:]
        v_new_fft_width_imag = ds.variables['fft_imag'][:]
        flow_mask = ds.variables['flow_mask'][:]

    # Compute magnitude of the complex Fourier coefficients
    v_new_fft_time_magnitude = np.sqrt(v_new_fft_time_real**2 + v_new_fft_time_imag**2)
    v_new_fft_height_magnitude = np.sqrt(v_new_fft_height_real**2 + v_new_fft_height_imag**2)
    v_new_fft_width_magnitude = np.sqrt(v_new_fft_width_real**2 + v_new_fft_width_imag**2)

    # Compute the energy spectrum for spatial dimensions (wave number)
    energy_wave_number_height = np.sum(v_new_fft_height_magnitude**2 * flow_mask, axis=(0, 2))/2
    energy_wave_number_height = energy_wave_number_height / np.sum(flow_mask)  # normalize by the number of non-masked elements

    energy_wave_number_width = np.sum(v_new_fft_width_magnitude**2 * flow_mask, axis=(0, 1))/2
    energy_wave_number_width = energy_wave_number_width / np.sum(flow_mask)  # normalize by the number of non-masked elements

    # Compute the energy spectrum for time dimension (frequency)
    energy_frequency = np.sum(v_new_fft_time_magnitude**2 * flow_mask, axis=(1, 2))/2
    energy_frequency = energy_frequency / np.sum(flow_mask)  # normalize by the number of non-masked elements

    # Ensure energy spectra have positive values for log scale
    energy_wave_number_height = np.clip(energy_wave_number_height, 1e-10, None)
    energy_wave_number_width = np.clip(energy_wave_number_width, 1e-10, None)
    energy_frequency = np.clip(energy_frequency, 1e-10, None)

    return energy_wave_number_height, energy_wave_number_width, energy_frequency

# Linear model for fitting
def linear_model(x, a, b):
    return a * x + b

# Function to save energy spectra to NetCDF files
def save_energy_spectra(frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width,filename):
    # Save time FFT results
    with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
        ds.createDimension('frequency', len(frequencies))
        freq_var = ds.createVariable('frequency', 'f4', ('frequency',))
        energy_var = ds.createVariable('energy', 'f4', ('frequency',))
        freq_var[:] = frequencies
        energy_var[:] = energy_frequency


# Function to plot energy spectra with linear fit and reference line
def plot_energy_spectra(energy_wave_number_height, energy_wave_number_width, energy_frequency):
    plt.figure(figsize=(21, 6))

    # Plot energy vs. frequency (from time FFT)
    plt.subplot(1, 3, 1)
    timestep = 0.1
    frequencies = np.fft.fftfreq(energy_frequency.shape[0], d=timestep)
    frequencies = frequencies[frequencies > 0]  # consider only positive frequencies
    frequencies = frequencies / 6  # convert to 1/10 minutes
    energy_frequency = energy_frequency[1:frequencies.size + 1]
    plt.plot(frequencies, energy_frequency, label='Energy Spectrum')

    # Fit linear model
    log_frequencies = np.log(frequencies)
    log_energy_frequency = np.log(energy_frequency)
    popt, _ = curve_fit(linear_model, log_frequencies, log_energy_frequency)
    plt.plot(frequencies, np.exp(linear_model(log_frequencies, *popt)), 'r--', label=f'Fit: log(y) = {popt[0]:.2f} * log(x) + {popt[1]:.2f}')

    # Reference line with slope -5/3 passing through midpoint
    #mid_index = len(frequencies) // 2
    mid_index=np.searchsorted(log_frequencies,(log_frequencies[0]+log_frequencies[-1])/2)
    mid_x = log_frequencies[mid_index]
    #mid_y = log_energy_frequency[mid_index]
    mid_y = linear_model(log_frequencies, *popt)[mid_index]
    #mid_x=(log_frequencies[0]+log_frequencies[-1])/2
    #mid_y=(log_energy_frequency[0]+log_energy_frequency[-1])/2

    slope = -5/3
    intercept = mid_y - slope * mid_x
    plt.plot(frequencies, np.exp(linear_model(log_frequencies, slope, intercept)), 'g--', label=f'Reference: slope = -5/3')

    plt.yscale('log')
    plt.xscale('log')
    plt.title('Energy vs. Frequency (Time FFT)')
    plt.xlabel('Frequency (1/10 minute)')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Plot energy vs. wave number (from height FFT)
    plt.subplot(1, 3, 2)
    heightstep = 1
    wave_numbers_height = np.fft.fftfreq(energy_wave_number_height.shape[0], d=heightstep)
    wave_numbers_height = wave_numbers_height[wave_numbers_height > 0]  # consider only positive frequencies
    energy_wave_number_height = energy_wave_number_height[1:wave_numbers_height.size + 1]
    plt.plot(wave_numbers_height, energy_wave_number_height, label='Energy Spectrum')

    # Fit linear model
    log_wave_numbers_height = np.log(wave_numbers_height)
    log_energy_wave_number_height = np.log(energy_wave_number_height)
    popt, _ = curve_fit(linear_model, log_wave_numbers_height, log_energy_wave_number_height)
    plt.plot(wave_numbers_height, np.exp(linear_model(log_wave_numbers_height, *popt)), 'r--', label=f'Fit: log(y) = {popt[0]:.2f} * log(x) + {popt[1]:.2f}')

    # Reference line with slope -5/3 passing through midpoint
    mid_index = len(wave_numbers_height) // 2
    mid_x = log_wave_numbers_height[mid_index]
    mid_y = log_energy_wave_number_height[mid_index]
    intercept = mid_y - slope * mid_x
    plt.plot(wave_numbers_height, np.exp(linear_model(log_wave_numbers_height, slope, intercept)), 'g--', label=f'Reference: slope = -5/3')

    plt.yscale('log')
    plt.xscale('log')
    plt.title('Energy vs. Wave Number (Height FFT)')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Plot energy vs. wave number (from width FFT)
    plt.subplot(1, 3, 3)
    widthstep = 1
    wave_numbers_width = np.fft.fftfreq(energy_wave_number_width.shape[0], d=widthstep)
    wave_numbers_width = wave_numbers_width[wave_numbers_width > 0]  # consider only positive frequencies
    energy_wave_number_width = energy_wave_number_width[1:wave_numbers_width.size + 1]
    plt.plot(wave_numbers_width, energy_wave_number_width, label='Energy Spectrum')

    # Fit linear model
    log_wave_numbers_width = np.log(wave_numbers_width)
    log_energy_wave_number_width = np.log(energy_wave_number_width)
    popt, _ = curve_fit(linear_model, log_wave_numbers_width, log_energy_wave_number_width)
    plt.plot(wave_numbers_width, np.exp(linear_model(log_wave_numbers_width, *popt)), 'r--', label=f'Fit: log(y) = {popt[0]:.2f} * log(x) + {popt[1]:.2f}')

    # Reference line with slope -5/3 passing through midpoint
    mid_index = len(wave_numbers_width) // 2
    mid_x = log_wave_numbers_width[mid_index]
    mid_y = log_energy_wave_number_width[mid_index]
    intercept = mid_y - slope * mid_x
    plt.plot(wave_numbers_width, np.exp(linear_model(log_wave_numbers_width, slope, intercept)), 'g--', label=f'Reference: slope = -5/3')

    plt.yscale('log')
    plt.xscale('log')
    plt.title('Energy vs. Wave Number (Width FFT)')
    plt.xlabel('Wave Number')
    plt.ylabel('Energy')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width


# Compute energy spectra considering the flow mask
energy_wave_number_height, energy_wave_number_width, energy_frequency = compute_energy_spectra_with_mask(input_nc_file1, input_nc_file2, input_nc_file3)

# Plot energy spectra and save results to NetCDF files
frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width = plot_energy_spectra(
    energy_wave_number_height, energy_wave_number_width, energy_frequency)

save_energy_spectra(frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width,filename)

input_nc_file1 = inputdir+'u_new_fft_time.nc'
input_nc_file2 = inputdir+'u_new_fft_height.nc'
input_nc_file3 = inputdir+'u_new_fft_width.nc'
filename=outputdir+'u_energy_time.nc'

# Compute energy spectra considering the flow mask
energy_wave_number_height, energy_wave_number_width, energy_frequency = compute_energy_spectra_with_mask(input_nc_file1, input_nc_file2, input_nc_file3)

# Plot energy spectra and save results to NetCDF files
frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width = plot_energy_spectra(
    energy_wave_number_height, energy_wave_number_width, energy_frequency)

save_energy_spectra(frequencies, energy_frequency, wave_numbers_height, energy_wave_number_height, wave_numbers_width, energy_wave_number_width,filename)

