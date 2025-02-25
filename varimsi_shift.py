
import numpy as np
from netCDF4 import Dataset

# -------------------------------
# 파일 경로 및 데이터 설정
# -------------------------------
inputdir = '/home/teamai/data/diff1000temptep240/just/'
nc_file = inputdir + 'v_new_data.nc'
output_nc_file = inputdir + 'u_v_shifted_time_full.nc'

boxstart = 214
beforebox = boxstart
spongebuf = 25
spinup = 1000  # spinup 1000개는 그대로 유지 (v_new = v_raw[spinup:])
# phase shift 시작/끝 값
phase_shift_value1 = shiftphase1
phase_shift_value2 = shiftphase2

target_freq = targetfreq  # 목표 주파수
bandwidth = 0.002      # 대역폭

window_size = 1000     # 윈도우 크기
half = window_size // 2  # 500

# -------------------------------
# NetCDF 파일에서 데이터 읽기
# -------------------------------
dataset = Dataset(nc_file, 'r')

# 영역 지정하여 변수 읽기
v_raw = dataset.variables['v_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
u_raw = dataset.variables['u_new'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
flow_mask = dataset.variables['flow_mask'][:, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]

# spinup 이후의 데이터 사용 (예: 총 30000포인트라면 v_new.shape[0] = 30000 - 1000 = 29000)
v_new = v_raw[spinup:]
u_new = u_raw[spinup:]
flow_mask_new = flow_mask[spinup:]
T_total, height, width = v_new.shape  # T_total = 29000 (예)

# 최종 출력 포인트 수는 원본 v_new와 동일하게 29000포인트로 만듭니다.
T_out = T_total

# FFT를 위한 시간축 샘플 간격 (원래 코드에서는 d=4)
d = 4
# window_size에 대한 주파수 배열 (윈도우마다 동일)
freqs = np.fft.fftfreq(window_size, d=d)
# 타겟 주파수 대역에 해당하는 인덱스 (window마다 동일)
band_indices = np.where((freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth))[0]

# 결과 저장 배열
v_shifted = np.empty((T_out, height, width), dtype=np.float32)
u_shifted = np.empty((T_out, height, width), dtype=np.float32)

# -------------------------------
# 각 시간 포인트별로 phase shift 적용
# -------------------------------
for j in range(T_out):
    # j에 따른 phase shift 선형 보간 (j=0이면 phase_shift_value1, j=T_out-1이면 phase_shift_value2)
    current_phase = phase_shift_value1 + (phase_shift_value2 - phase_shift_value1) * (j / (T_out - 1))
    
    # 경계 처리:
    if j < half:
        # 처음 500포인트: 첫 번째 윈도우 (v_new[0:1000])
        window_v = v_new[0:window_size]
        window_u = u_new[0:window_size]
    elif j >= T_out - half:
        # 마지막 500포인트: 마지막 윈도우 (v_new[-1000:])
        window_v = v_new[-window_size:]
        window_u = u_new[-window_size:]
    else:
        # 중간: 윈도우 중앙이 j가 되도록 선택 (즉, v_new[j-half : j-half+window_size])
        window_v = v_new[j - half : j - half + window_size]
        window_u = u_new[j - half : j - half + window_size]
    
    # FFT 수행 (시간축)
    fft_v = np.fft.fft(window_v, axis=0)
    fft_u = np.fft.fft(window_u, axis=0)
    
    # 타겟 주파수 대역에 phase shift 적용
    fft_v[band_indices, :, :] *= np.exp(1j * current_phase)
    fft_u[band_indices, :, :] *= np.exp(1j * current_phase)
    
    # 역 FFT 수행하여 시간 도메인 데이터 복원 (실수부 사용)
    ifft_v = np.fft.ifft(fft_v, axis=0).real
    ifft_u = np.fft.ifft(fft_u, axis=0).real
    
    # 대표값은 윈도우의 중앙 (index = half)
    v_shifted[j, :, :] = ifft_v[half]
    u_shifted[j, :, :] = ifft_u[half]
    if j % 100 == 0:
        print(j)

# -------------------------------
# 결과를 NetCDF 파일에 저장
# -------------------------------
new_dataset = Dataset(output_nc_file, 'w', format='NETCDF4')
new_dataset.createDimension('time', T_out)
new_dataset.createDimension('height', height)
new_dataset.createDimension('width', width)

v_var = new_dataset.createVariable('v_shifted', np.float32, ('time', 'height', 'width'))
u_var = new_dataset.createVariable('u_shifted', np.float32, ('time', 'height', 'width'))
mask_var = new_dataset.createVariable('flow_mask', np.float32, ('time', 'height', 'width'))

v_var[:] = v_shifted
u_var[:] = u_shifted
# flow_mask는 그대로 시간 차원에 맞춰서 사용 (경계 처리 없이)
mask_var[:] = flow_mask_new[:T_out]

dataset.close()
new_dataset.close()

print(f"Phase shift가 적용된 데이터가 {T_out}포인트(경계 500포인트는 첫/마지막 결과 사용)로 '{output_nc_file}'에 저장되었습니다.")

