import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# 입력 및 출력 파일 경로 설정
inputdir = '/home/teamai/data/diff1000ep240/just/'
outputdir = inputdir
nc_file = inputdir + 'v_new_data.nc'
output_nc_file = outputdir + 'u_v_shifted_time.nc'

# 데이터 설정
#boxstart = 210
boxstart = 214
#beforebox = boxstart - 20
beforebox = boxstart
spongebuf = 25
spinup = 1000
forplot = 1700
phase_shift_value = shiftphase  # 위상 shift 값 설정
target_freq = 0.00568 # 목표로 하는 특정 wavenumber
#target_freq = 0.0067 # 목표로 하는 특정 wavenumber
bandwidth=0.002

# NetCDF 파일 열기
dataset = Dataset(nc_file, 'r')

# 필요한 변수 읽기 (spinup 이후, 특정 구간의 데이터만 추출)
v_raw = dataset.variables['v_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
u_raw = dataset.variables['u_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
flow_mask = dataset.variables['flow_mask'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
time_raw, height, width = v_raw.shape
u_new = u_raw[spinup:]
v_new = v_raw[spinup:]

# time, height, width의 크기 확인
time, height, width = v_new.shape
def apply_phase_shift(data, phase_shift_value, target_freq, bandwidth):
    # Fourier 변환 (time 방향)
    data_hat = np.fft.fft(data, axis=0)

    # 주파수 계산 (Fourier 변환의 주파수 성분)
    k = np.fft.fftfreq(time, d=4)  # `d`는 샘플 간격 (여기선 4)

    # 특정 주파수 대역(bandwidth) 지정
    band_indices = np.where((k >= target_freq - bandwidth) & (k <= target_freq + bandwidth))[0]

    # 위상 shift 적용
    phase_shift = np.ones_like(data_hat, dtype=complex)  # 기본적으로 shift 없음
    for idx in band_indices:
        phase_shift[idx, :, :] = np.exp(1j * phase_shift_value)  # 특정 주파수 대역에만 위상 shift 적용

    # 주파수 도메인에서 위상 shift 적용
    data_hat_shifted = data_hat * phase_shift

    # 역 Fourier 변환
    data_shifted = np.fft.ifft(data_hat_shifted, axis=0)
    data_shifted = np.real(data_shifted)  # 실수부만 사용

    return data_shifted, data_hat, data_hat_shifted


# 각 time step에 대해 u_new와 v_new에 위상 shift 적용
u_shifted = np.empty_like(u_new)
v_shifted = np.empty_like(v_new)

# Fourier 변환된 결과를 저장할 리스트
print(u_shifted.shape)
print(u_shifted[0].shape)

u_shifted, u_hat, u_hat_shifted = apply_phase_shift(u_new, phase_shift_value, target_freq,bandwidth)

# v_new에 대해 shift 적용 및 에너지 계산
v_shifted, v_hat, v_hat_shifted = apply_phase_shift(v_new, phase_shift_value, target_freq,bandwidth)

u_raw[spinup:]=u_shifted[:]
v_raw[spinup:]=v_shifted[:]

# 새로운 NetCDF 파일 생성 및 저장
new_dataset = Dataset(output_nc_file, 'w', format='NETCDF4')

# 차원 생성 (time, height, width)
new_dataset.createDimension('time', time_raw)
new_dataset.createDimension('height', height)
new_dataset.createDimension('width', width)

# 변수 생성
v_shifted_var = new_dataset.createVariable('v_shifted', np.float32, ('time', 'height', 'width'))
u_shifted_var = new_dataset.createVariable('u_shifted', np.float32, ('time', 'height', 'width'))
flow_mask_var = new_dataset.createVariable('flow_mask', np.float32, ('time', 'height', 'width'))

# 데이터 저장
v_shifted_var[:] = v_raw
u_shifted_var[:] = u_raw
flow_mask_var[:] = flow_mask


# 파일 닫기
dataset.close()
new_dataset.close()

print(f"u_new와 v_new에 대한 위상 shift가 적용된 데이터가 '{output_nc_file}' 파일에 저장되었습니다.")

