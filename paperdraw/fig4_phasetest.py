#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용
import matplotlib.pyplot as plt

# 파일 경로 설정
inputdir   = '/home/teamai/data/diff1000ep240/'
ref_file   = inputdir + 'just/v_new_data.nc'
shiftdir1  = inputdir + 'shift_np.pi_4/'
shiftdir2  = inputdir + 'shift_np.pi_2/'
shiftdir3  = inputdir + 'shift_3*np.pi_4/'
file_shift1 = shiftdir1 + 'v_new_data.nc'
file_shift2 = shiftdir2 + 'v_new_data.nc'
file_shift3 = shiftdir3 + 'v_new_data.nc'
pngfile = 'diff1000ep240_shiftphase_subplots.png'

# Reference values for each dataset
ref_value1 = np.pi/4
ref_value2 = np.pi/2
ref_value3 = 3*np.pi/4

# 설정 값
target_freq = 0.00568   # 비교하고자 하는 주파수
boxstart = 214
beforebox = boxstart
spongebuf = 25
spinup = 1000
timewindow = 1000

# 데이터 로딩 함수 (영역 슬라이싱 적용)
def load_data(nc_file, start_index, end_index):
    dataset = Dataset(nc_file, 'r')
    u_new = dataset.variables['u_new'][start_index:end_index, 3+spongebuf : -spongebuf-3, 5:beforebox]
    v_new = dataset.variables['v_new'][start_index:end_index, 3+spongebuf : -spongebuf-3, 5:beforebox]
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

# 두 실험 간 위상 차이를 계산하는 함수 (기준: ref_file, 비교: shift_file)
def process_pair(ref_file, shift_file):
    with Dataset(ref_file, 'r') as ds:
        timesize = ds.variables['u_new'].shape[0] - spinup
    mean_u_values = []
    mean_v_values = []
    time_indices = []
    for timeindex in range(0, timesize - timewindow, 500):
        # 기준 데이터 로드
        u_ref, v_ref = load_data(ref_file, spinup+timeindex, spinup+timeindex+timewindow)
        # shift 데이터 로드
        u_shift, v_shift = load_data(shift_file, spinup+timeindex, spinup+timeindex+timewindow)
        # 위상 추출
        phase_ref_u = extract_phase(u_ref, target_freq)
        phase_ref_v = extract_phase(v_ref, target_freq)
        phase_shift_u = extract_phase(u_shift, target_freq)
        phase_shift_v = extract_phase(v_shift, target_freq)
        # 위상 차이 계산 (shift - ref)
        phase_diff_u = phase_shift_u - phase_ref_u
        phase_diff_v = phase_shift_v - phase_ref_v
        # 위상 차이를 [-π, π] 범위로 조정
        phase_diff_u = np.where(phase_diff_u > np.pi, phase_diff_u - 2*np.pi, phase_diff_u)
        phase_diff_u = np.where(phase_diff_u < -np.pi, phase_diff_u + 2*np.pi, phase_diff_u)
        phase_diff_v = np.where(phase_diff_v > np.pi, phase_diff_v - 2*np.pi, phase_diff_v)
        phase_diff_v = np.where(phase_diff_v < -np.pi, phase_diff_v + 2*np.pi, phase_diff_v)
        # 평균 값 계산
        mean_u = np.mean(phase_diff_u)
        mean_v = np.mean(phase_diff_v)
        mean_u_values.append(mean_u)
        mean_v_values.append(mean_v)
        time_indices.append(timeindex + timewindow//2)
    return time_indices, mean_u_values, mean_v_values

# 각 쌍에 대해 데이터 처리
time_indices1, mean_u_values1, mean_v_values1 = process_pair(ref_file, file_shift1)
time_indices2, mean_u_values2, mean_v_values2 = process_pair(ref_file, file_shift2)
time_indices3, mean_u_values3, mean_v_values3 = process_pair(ref_file, file_shift3)

# 서브플롯 생성 (1행 3열)
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True, sharey=True)

# 각 서브플롯에 대해 플롯 그리기
def plot_subplot(ax, time_indices, mean_u, mean_v, ref_value, title):
    ax.plot(time_indices, mean_u, label='Mean Phase Diff (u)', color='blue', linewidth=2)
    ax.plot(time_indices, mean_v, label='Mean Phase Diff (v)', color='red', linewidth=2)
    # ref_value에 따라 적절한 문자열 결정
    if np.isclose(ref_value, np.pi/4):
        ref_label = r'$\pi/4$'
    elif np.isclose(ref_value, np.pi/2):
        ref_label = r'$\pi/2$'
    elif np.isclose(ref_value, 3*np.pi/4):
        ref_label = r'$3\pi/4$'
    else:
        ref_label = f'{ref_value}'
    # 참조선 추가 (점선)
    ax.axhline(y=ref_value, color='green', linestyle=':', linewidth=2, label=r'Ref: ' +ref_label)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 3)
    ax.set_title(title, fontsize=16, pad=12)
    if np.isclose(ref_value, 3*np.pi/4):
        ax.set_xlabel('Time step', fontsize=14, labelpad=8)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 범례를 오른쪽 상단에 고정
    ax.legend(fontsize=12, loc='lower right', frameon=True)

plot_subplot(axs[0], time_indices1, mean_u_values1, mean_v_values1, ref_value1, r'Shift: $\pi/4$')
plot_subplot(axs[1], time_indices2, mean_u_values2, mean_v_values2, ref_value2, r'Shift: $\pi/2$')
plot_subplot(axs[2], time_indices3, mean_u_values3, mean_v_values3, ref_value3, r'Shift: $3\pi/4$')

# 전체 y축 레이블 추가 (공통)
fig.text(0.06, 0.5, 'Mean Phase Diff (radians)', va='center', rotation='vertical', fontsize=16)
# 전체 타이틀 (옵션)
fig.suptitle('Temporal Variation of Phase Difference', fontsize=18, y=0.98)

plt.tight_layout(rect=[0.08, 0.08, 0.98, 0.92])
plt.savefig(pngfile, dpi=300, bbox_inches='tight')
plt.close()

