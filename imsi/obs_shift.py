import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# 입력 및 출력 파일 경로 설정
inputdir = './obs_data/before/'
outputdir = './obs_data/adjust/'
nc_file = inputdir + 'v_new_data.nc'
output_nc_file = outputdir + 'v_shifted.nc'

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
time, height, width = v_new.shape
# Fourier 변환 및 특정 wavenumber에 대한 위상 shift 적용 함수
def apply_phase_shift(data, phase_shift_value, target_wavenumber):
    # width 방향에 대해 Fourier 변환
    data_hat = np.fft.fft(data, axis=1)  # axis=2는 width 방향

    # 주파수 성분 계산 (Fourier 변환의 wavenumber 성분)
    k = np.fft.fftfreq(width)  # width 방향의 wavenumber 성분

    # 특정 wavenumber에 해당하는 주파수 성분을 찾아 위상 shift 적용
    phase_shift = np.ones_like(k, dtype=complex)  # 기본적으로 shift 없음
    target_index = np.argmin(np.abs(k - target_wavenumber))  # 특정 wavenumber에 가까운 주파수 찾기
    phase_shift[target_index] = np.exp(1j * phase_shift_value)  # 선택한 wavenumber에만 위상 shift 적용

    # 주파수 도메인에서 위상 shift 적용
    data_hat_shifted = data_hat * phase_shift[np.newaxis, np.newaxis, :]

    # 역 Fourier 변환으로 공간 영역으로 변환
    data_shifted = np.fft.ifft(data_hat_shifted, axis=2)
    data_shifted = np.real(data_shifted)  # 실수부만 사용

    return data_shifted, data_hat, data_hat_shifted

# 각 time step에 대해 u_new와 v_new에 위상 shift 적용
u_shifted = np.empty_like(u_new)
v_shifted = np.empty_like(v_new)

# Fourier 변환된 결과를 저장할 리스트

for t in range(time):
    # u_new에 대해 shift 적용 및 에너지 계산
    u_shifted[t], u_hat, u_hat_shifted = apply_phase_shift(u_new[t], phase_shift_value, target_wavenumber)

    # v_new에 대해 shift 적용 및 에너지 계산
    v_shifted[t], v_hat, v_hat_shifted = apply_phase_shift(v_new[t], phase_shift_value, target_wavenumber)


# 새로운 NetCDF 파일 생성 및 저장
new_dataset = Dataset(output_nc_file, 'w', format='NETCDF4')

# 차원 생성 (time, height, width)
new_dataset.createDimension('time', time)
new_dataset.createDimension('height', height)
new_dataset.createDimension('width', width)

# 변수 생성
v_shifted_var = new_dataset.createVariable('v_shifted', np.float32, ('time', 'height', 'width'))
u_shifted_var = new_dataset.createVariable('u_shifted', np.float32, ('time', 'height', 'width'))
flow_mask_var = new_dataset.createVariable('flow_mask', np.float32, ('time', 'height', 'width'))

# 데이터 저장
v_shifted_var[:] = v_shifted
u_shifted_var[:] = u_shifted
flow_mask_var[:] = flow_mask


# 파일 닫기
dataset.close()
new_dataset.close()

print(f"u_new와 v_new에 대한 위상 shift가 적용된 데이터가 '{output_nc_file}' 파일에 저장되었습니다.")

