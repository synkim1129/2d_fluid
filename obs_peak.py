import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 입력 및 출력 파일 경로 설정
inputdir = './obs_data/before/'
outputdir = './obs_data/peak/'
nc_file = inputdir + 'v_new_data.nc'
output_nc_file = outputdir + 'v_add_peak_async.nc'

# 데이터 설정
boxstart = 214
beforebox = boxstart - 10
spongebuf = 25
spinup = 700
forplot = 1700
phase_shift_value = np.pi / 4  # 위상 shift 값 설정
target_wavenumber = 0.5  # 목표로 하는 특정 wavenumber

# NetCDF 파일 열기
dataset = Dataset(nc_file, 'r')

# 필요한 변수 읽기 (spinup 이후, 특정 구간의 데이터만 추출)
v_new = dataset.variables['v_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
u_new = dataset.variables['u_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
flow_mask = dataset.variables['flow_mask'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]

# time, height, width의 크기 확인
time_steps, height, weight = v_new.shape

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

# 시간 축(시간 방향)으로 FFT를 수행하고 특정 주파수를 증폭하는 함수
def amplify_frequency(data, target_frequency, time_steps, label=None, amplification_factor=None, amplified_value=None):
    """
    시간 축(시간 방향)에 대해 Fourier 변환을 수행하고 특정 주파수의 에너지를 증폭합니다.

    Parameters:
    - data: 입력 데이터 (예: u_new, v_new 등)
    - amplification_factor: 증폭할 배수 (예: 2.0이면 두 배 증폭)
    - target_frequency: 증폭할 주파수 (정규화된 주파수)
    - time_steps: 총 시간 스텝 수 (샘플 수)

    Returns:
    - data_amplified: 시간 영역으로 복원된 증폭된 데이터
    - data_hat: 원래의 주파수 도메인 데이터 (FFT 결과)
    - data_hat_amplified: 주파수 증폭 후의 데이터
    """
    # 시간 축으로 Fourier 변환
    data_hat = np.fft.fft(data, axis=0)  # axis=0은 시간 방향

    # 주파수 성분 계산 (정규화된 주파수 사용)
    freqs = np.fft.fftfreq(time_steps)

    # 특정 주파수에 해당하는 인덱스 찾기
    target_index = np.argmin(np.abs(freqs - target_frequency))

    # 해당 주파수 성분의 에너지 증폭
    data_hat_amplified = np.copy(data_hat)
    
    if amplification_factor is not None and amplified_value is None:
        data_hat_amplified[target_index, :, :] *= amplification_factor
    elif amplified_value is not None and amplification_factor is None:
        data_hat_amplified[target_index, :, :] = amplified_value
    else:
        raise ValueError("amplification_factor와 amplified_value 중 하나만 지정해야 합니다.")

    # 역 Fourier 변환으로 시간 영역으로 복원
    data_amplified_complex = np.fft.ifft(data_hat_amplified, axis=0)
    data_amplified = np.real(data_amplified_complex)  # 실수부만 반환
    
    energy_frequency = np.sum((np.abs(data_hat) ** 2) * flow_mask, axis=(1, 2))
    energy_frequency_amplified = np.sum((np.abs(data_hat_amplified) ** 2) * flow_mask, axis=(1, 2))
    
    positive_index = freqs > 0
    freqs = freqs[positive_index]
    energy_frequency = energy_frequency[positive_index]
    energy_frequency_amplified = energy_frequency_amplified[positive_index]
    
    peaks_frequencies, peaks_energies = find_all_peaks(freqs, energy_frequency)
    peaks_frequencies_amplified, peaks_energies_amplified = find_all_peaks(freqs, energy_frequency_amplified)
    
    
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(freqs, energy_frequency)
    plt.title(f"Energy Spectrum for {label} (Before Amplification)")
    plt.xlabel("Frequency")
    plt.ylabel("Energy")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.scatter(peaks_frequencies, peaks_energies, c="red")
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs, energy_frequency_amplified)
    plt.title(f"Energy Spectrum for {label} (After Amplification)")
    plt.xlabel("Frequency")
    plt.ylabel("Energy")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.scatter(peaks_frequencies_amplified, peaks_energies_amplified, c="red")
    
    plt.tight_layout()
    plt.savefig(f'./obs_data/adjust/energy_spectrum_async_{label}.png')

    return data_amplified

# 예시: u_new와 v_new에 대해 시간 축으로 FFT와 주파수 증폭 적용

# target_frequency = 0.16139534883720932  # 증폭할 주파수 (정규화된 값)
target_frequency = 0.2
amplified_value = 1000

# v_new에 대해 주파수 증폭 적용
v_amplified = amplify_frequency(
    v_new, target_frequency, time_steps, label='v', amplified_value=amplified_value
)

# u_new에 대해 주파수 증폭 적용
u_amplified = amplify_frequency(
    u_new, target_frequency, time_steps, label='u', amplified_value=amplified_value
)

new_dataset = Dataset(output_nc_file, 'w', format='NETCDF4')

# 차원 생성 (time, height, width)
new_dataset.createDimension('time', time_steps)
new_dataset.createDimension('height', height)
new_dataset.createDimension('width', weight)

# 변수 생성
v_amplified_var = new_dataset.createVariable('v_amplified', np.float32, ('time', 'height', 'width'))
u_amplified_var = new_dataset.createVariable('u_amplified', np.float32, ('time', 'height', 'width'))
flow_mask_var = new_dataset.createVariable('flow_mask', np.float32, ('time', 'height', 'width'))

# 데이터 저장
v_amplified_var[:] = v_amplified
u_amplified_var[:] = u_amplified
flow_mask_var[:] = flow_mask

# 파일 닫기
dataset.close()
new_dataset.close()