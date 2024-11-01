import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# File paths
inputdir = '/home/binpyo/data/before/240722/'
outputdir = '/home/binpyo/data/before/240722/'
nc_file_u = inputdir + 'u_energy_time.nc'
nc_file_v = inputdir + 'v_energy_time.nc'
new_nc_file_u = outputdir + 'u_adjustment.nc'
new_nc_file_v = outputdir + 'v_adjustment.nc'
windowsize = 100

# Function to read energy vs frequency data from NetCDF file
def read_energy_vs_frequency(nc_file):
    with nc.Dataset(nc_file, 'r') as ds:
        frequencies = ds.variables['frequency'][:]
        energy_frequency = ds.variables['energy'][:]
    return frequencies, energy_frequency

# Function to calculate the slope using a sliding window
def calculate_slope(frequencies, energy_frequency, window_size):
    log_frequencies = np.log(frequencies)
    log_energy_frequency = np.log(energy_frequency)
    slopes = []

    for i in range(len(log_frequencies) - window_size + 1):
        x_window = log_frequencies[i:i + window_size]
        y_window = log_energy_frequency[i:i + window_size]
        slope, _ = np.polyfit(x_window, y_window, 1)
        slopes.append(slope)

    return slopes, log_frequencies, log_energy_frequency

# Function to find the first steep section and modify energy
def modify_energy(frequencies, energy_frequency, slopes, log_frequencies, log_energy_frequency):
    reference_slope = -5 / 3

    # Find the first point where the slope is steeper than -5/3
    start_index = next(i for i, slope in enumerate(slopes) if slope < reference_slope) + int((windowsize - 1)/2)

    # Modify the energy from the first steep point onward with slope -5/3
    log_frequencies_to_modify = np.log(frequencies[start_index:])
    modified_log_energy = reference_slope * log_frequencies_to_modify + (log_energy_frequency[start_index] - reference_slope * log_frequencies[start_index])
    modified_energy_frequency_raw = np.exp(modified_log_energy)

    # Update the energy array
    modified_energy_frequency=np.copy(energy_frequency)
    modified_energy_frequency[start_index:] = modified_energy_frequency_raw
    return modified_energy_frequency

# Function to save the modified energy vs frequency data to a new NetCDF file
def save_modified_energy_vs_frequency(frequencies, energy_frequency, adjustment_energy,nc_file):
    with nc.Dataset(nc_file, 'w', format='NETCDF4') as ds:
        ds.createDimension('frequency', len(frequencies))
        freq_var = ds.createVariable('frequency', 'f4', ('frequency',))
        energy_var = ds.createVariable('energy', 'f4', ('frequency',))
        adjustment_var = ds.createVariable('adjustment_energy', 'f4', ('frequency',))
        freq_var[:] = frequencies
        energy_var[:] = energy_frequency
        adjustment_var[:] = adjustment_energy

# Function to plot old and new energy vs. frequency
def plot_old_and_new_energy(frequencies, old_energy, new_energy):
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(frequencies), np.log(old_energy), label='Old Log(Energy) vs Log(Frequency)')
    plt.plot(np.log(frequencies), np.log(new_energy), label='New Log(Energy) vs Log(Frequency)')
    plt.xlabel('Log(Frequency)')
    plt.ylabel('Log(Energy)')
    plt.title('Old and New Log-Log Plot of Energy vs Frequency')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# Main processing for a given nc_file
def process_energy_file(nc_file, new_nc_file):
    frequencies, energy_frequency = read_energy_vs_frequency(nc_file)
    slopes, log_frequencies, log_energy_frequency = calculate_slope(frequencies, energy_frequency, windowsize)
    modified_energy_frequency = modify_energy(frequencies, energy_frequency, slopes, log_frequencies, log_energy_frequency)
    adjustment_energy=modified_energy_frequency-energy_frequency
    save_modified_energy_vs_frequency(frequencies, modified_energy_frequency,adjustment_energy,new_nc_file)
    plot_old_and_new_energy(frequencies, energy_frequency, modified_energy_frequency)

# Process both u and v energy files
process_energy_file(nc_file_u, new_nc_file_u)
process_energy_file(nc_file_v, new_nc_file_v)

