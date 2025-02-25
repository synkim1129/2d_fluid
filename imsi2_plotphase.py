import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용
import matplotlib.pyplot as plt

# 파일 경로 설정
inputdir = '/home/teamai/data/diff1000ep240/'
shiftdir= inputdir+'shiftname/'
file2 = inputdir + 'just/v_new_data.nc'
file3 = shiftdir + 'v_new_data.nc'
pngfile= shiftdir + 'diff1000ep240_shiftphase.png'

# 타겟 주파수 설정
target_freq = 0.00568  # 비교하고자 하는 주파수
#boxstart = 210
boxstart = 214
#beforebox = boxstart - 20
beforebox = boxstart
spongebuf = 25
spinup = 1000
timewindow = 1000

# 데이터 로딩 함수
def load_data(nc_file, start_index, end_index):
    dataset = Dataset(nc_file, 'r')
    u_new = dataset.variables['u_new'][start_index:end_index, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
    v_new = dataset.variables['v_new'][start_index:end_index, 3 + spongebuf : -spongebuf - 3, 5 : beforebox]
    dataset.close()
    return u_new, v_new

# Fourier 변환 및 위상 추출 함수
def extract_phase(data, target_freq):
    time, height, width = data.shape
    data_hat = np.fft.fft(data, axis=0)
    k = np.fft.fftfreq(time, d=4)
    target_index = np.argmin(np.abs(k - target_freq))
    phase = np.angle(data_hat[target_index, :, :])
    return phase

# 총 데이터 크기 확인
with Dataset(file2, 'r') as dataset:
    timesize = dataset.variables['u_new'].shape[0] - spinup

mean_u_values = []
mean_v_values = []
time_indices = []

for timeindex in range(0, timesize - timewindow, 500):
    # 데이터 로드
    u_new_exp2, v_new_exp2 = load_data(file2, spinup + timeindex, spinup + timeindex + timewindow)
    u_new_exp3, v_new_exp3 = load_data(file3, spinup + timeindex, spinup + timeindex + timewindow)

    # Fourier 변환 및 위상 추출
    phase_u_exp2 = extract_phase(u_new_exp2, target_freq)
    phase_u_exp3 = extract_phase(u_new_exp3, target_freq)
    phase_v_exp2 = extract_phase(v_new_exp2, target_freq)
    phase_v_exp3 = extract_phase(v_new_exp3, target_freq)

    # 위상 차이 계산
    phase_diff_u_23 = phase_u_exp3 - phase_u_exp2
    phase_diff_v_23 = phase_v_exp3 - phase_v_exp2

    # 위상 차이를 [-π, π] 범위로 조정
    phase_diff_u_23 = np.where(phase_diff_u_23 > np.pi, phase_diff_u_23 - 2 * np.pi, phase_diff_u_23)
    phase_diff_u_23 = np.where(phase_diff_u_23 < -np.pi, phase_diff_u_23 + 2 * np.pi, phase_diff_u_23)
    phase_diff_v_23 = np.where(phase_diff_v_23 > np.pi, phase_diff_v_23 - 2 * np.pi, phase_diff_v_23)
    phase_diff_v_23 = np.where(phase_diff_v_23 < -np.pi, phase_diff_v_23 + 2 * np.pi, phase_diff_v_23)

    # 평균 값 계산
    mean_u = np.mean(phase_diff_u_23)
    mean_v = np.mean(phase_diff_v_23)

    mean_u_values.append(mean_u)
    mean_v_values.append(mean_v)
    time_indices.append(timeindex + timewindow // 2)

# 결과 시각화
plt.figure(figsize=(15, 5))
plt.plot(time_indices, mean_u_values, label='Mean Phase Diff (u_new)', color='blue')
plt.plot(time_indices, mean_v_values, label='Mean Phase Diff (v_new)', color='red')
plt.xlabel('Time Index')
plt.ylabel('Mean Phase Diff (radians)')
plt.title('Mean Phase Difference vs Time (u_new & v_new)')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(pngfile, dpi=300, bbox_inches='tight') 
plt.show()

